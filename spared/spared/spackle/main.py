import os
import json
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
from spackle.utils import *
from spackle.model import GeneImputationModel
from lightning.pytorch import Trainer
from spackle.dataset import ImputationDataset
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import pathlib
import sys

# El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from datasets import datasets
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

## Set of auxiliary functions for model test and comparison
def get_imputation_results_from_trained_model(trainer, model, best_model_path, train_loader, val_loader, test_loader = None):
    """
    This function tests the incoming SpaCKLE completion model in all data splits available using PyTorch Lightning.

    Args:
        trainer (lightning.Trainer): PyTorch lightning trainer used for model training and testing.
        model (GeneImputationModel): Imputation model with loaded weights to test perfomance.
        best_model_path (str): Path to the checkpoints that will be tested.
        train_loader (torch.DataLoader): DataLoader of the train data split. 
        val_loader (torch.DataLoader): DataLoader of the val data split. 
        test_loader (torch.DataLoader, optional): If available, DataLoader of the test data split. 

    Return:
        train_model_imputation_metrics (dict): Evaluation metrics when testing the model on train split.
        val_model_imputation_metrics (dict): Evaluation metrics when testing the model on val split.
        test_model_imputation_metrics (dict): Returned if test data is provided, else is None. Evaluation metrics when testing the model on test split.
    """
    ## Results for imputation model
    train_model_imputation_metrics = trainer.test(model = model, dataloaders = train_loader, ckpt_path = best_model_path)[0]
    val_model_imputation_metrics = trainer.test(model = model, dataloaders = val_loader, ckpt_path = best_model_path)[0]
    test_model_imputation_metrics = None

    # Use test_split too if available
    if test_loader is not None:
        test_model_imputation_metrics = trainer.test(model = model, dataloaders = test_loader, ckpt_path = best_model_path)[0]

    return train_model_imputation_metrics, val_model_imputation_metrics, test_model_imputation_metrics

def get_complete_imputation_results(model, trainer, best_model_path, args, prob_tensor, device, prediction_layer, train_split, val_split, test_split = None):
    """
    This function gets the evaluation metrics of both the median filter and the trained model in all data splits available for comparison of both completion methods.

    Args:
        model (model): imputation model with loaded weights to test perfomance.
        trainer (lightning.Trainer): pytorch lightning trainer used for model training and testing.
        best_model_path (str): path to the checkpoints that will be tested.
        args (dict): dictionary with the values necessary for data processing.
        prob_tensor (torch.Tensor): vector with the masking probabilities for each gene. Shape: n_genes  
        device (torch.device): device in which tensors will be processed.
        train_split (ad.AnnData): adata of the train data split before being masked and imputed through median and trained model.
        val_split (ad.AnnData): adata of the val data split before being masked and imputed through median and trained model.
        test_split (ad.AnnData, optional): if available, adata of the test data split before being masked and imputed through median and trained model.

    Return:
        complete_imputation_results (dict): contains the evaluation metrics of the imputation through both methods (median and model) in all data splits available.
        train_split (ad.AnnData): updated train adata with the prediction layers included.
        val_split (ad.AnnData): updated val adata with the prediction layers included.
        test_split (ad.AnnData): if not None, updated test adata with the prediction layers included.

    """
    complete_imputation_results = {}
        
    train_split = mask_exp_matrix(adata = train_split, pred_layer = prediction_layer, mask_prob_tensor = prob_tensor, device = device)
    val_split = mask_exp_matrix(adata = val_split, pred_layer = prediction_layer, mask_prob_tensor = prob_tensor, device = device)
    
    if test_split is not None:
        test_split = mask_exp_matrix(adata = test_split, pred_layer = prediction_layer, mask_prob_tensor = prob_tensor, device = device)
        
    ## Prepare DataLoaders for testing on trained model
    train_data = ImputationDataset(train_split, args, 'train', prediction_layer, pre_masked = True)
    train_loader = DataLoader(train_data, batch_size=args["batch_size"], shuffle=False, pin_memory=False, drop_last=True, num_workers=args["num_workers"], generator=torch.Generator("cuda"))

    val_data = ImputationDataset(val_split, args, 'val', prediction_layer, pre_masked = True)
    val_loader = DataLoader(val_data, batch_size=args["batch_size"], shuffle=False, pin_memory=False, drop_last=True, num_workers=args["num_workers"], generator=torch.Generator("cuda"))
    test_loader = None
    if test_split is not None:
        test_data = ImputationDataset(test_split, args, 'test', prediction_layer, pre_masked = True)
        test_loader = DataLoader(test_data, batch_size=args["batch_size"], shuffle=False, pin_memory=False, drop_last=True, num_workers=args["num_workers"], generator=torch.Generator("cuda"))

    ## Results for trained model
    trained_model_results = get_imputation_results_from_trained_model(
        trainer, model, best_model_path, 
        train_loader, val_loader, test_loader = test_loader)

    # Build dictionary with results from median and model
    complete_imputation_results = {
        'train_spackle_results': trained_model_results[0],
        'val_spackle_results': trained_model_results[1]
        }
    
    if test_split is not None:
        complete_imputation_results['test_spackle_results'] = trained_model_results[2]
    
    return complete_imputation_results, train_split, val_split, test_split


def train_spackle(adata, device, save_path, prediction_layer, lr, train, get_performance, load_ckpt_path, args_dict, optimizer = "Adam", max_steps = 10000):
    """
    This function declares, trains and tests a new SpaCKLE model. The best model is saved in save_path. Alternatively, this function can receive the path to pretrained 
    checkpoints and only test the model to analyze the performance of SpaCKLE.

    Args:
        adata (ad.AnnData): Complete adata (with inner train, val, and test splits in `adata.obs`).
        device (torch.device): Device in which tensors will be processed.
        save_path (str): Directory to save the model file.
        prediction_layer (str): Layer in adata.layers where the gene expression matrix that will be completed is located. This matrix corresponds to the ground truth when training.
        lr (float): The learning rate for training the model.
        train (bool): If True, a new SpaCKLE model will be trained and tested, otherwise, the function will only test the pretrained model found in `load_ckpt_path`.
        get_performance (bool): If True, the function will calculate the final evaluation metrics and save them in a txt file in save_path.
        load_ckpt_path (str): Path to the checkpoints of a pretrained SpaCKLE model. This path should lead directly to the .ckpt file.
        args_dict (dict): A dictionary with the values needed for processing the data and building the model's architecture. For more information on the required keys, refer to the 
                          documentation of the function `get_args_dict()` in `spared.spackle.utils`.
        optimizer (str, optional): The name of the optimizer selected for the training process. Default = "Adam".
        max_steps (int, optional): Stop training after this number of steps. Default = 10000
    """
    # Check if test split is available
    test_data_available = True if 'test' in adata.obs['split'].unique() else False
    # Declare data splits
    train_split = adata[adata.obs['split']=='train']
    val_split = adata[adata.obs['split']=='val']
    test_split = adata[adata.obs['split']=='test'] if test_data_available else None

    # Prepare data and create dataloaders
    train_data = ImputationDataset(train_split, args_dict, 'train', prediction_layer)
    val_data = ImputationDataset(val_split, args_dict, 'val', prediction_layer)

    train_loader = DataLoader(train_data, batch_size=args_dict["batch_size"], shuffle=args_dict["shuffle"], pin_memory=False, drop_last=True, num_workers=args_dict["num_workers"], generator=torch.Generator("cuda"))
    val_loader = DataLoader(val_data, batch_size=args_dict["batch_size"], shuffle=args_dict["shuffle"], pin_memory=False, drop_last=True, num_workers=args_dict["num_workers"], generator=torch.Generator("cuda"))
    
    # Get masking probability tensor for training
    train_prob_tensor = get_mask_prob_tensor(args_dict["masking_method"], adata, args_dict["mask_prob"], args_dict["scale_factor"])
    # Get masking probability tensor for validating and testing with fixed method 'prob_median'
    val_test_prob_tensor = get_mask_prob_tensor(args_dict["masking_method"], adata, args_dict["mask_prob"], args_dict["scale_factor"])
    # Declare model
    vis_features_dim = 0
    model = GeneImputationModel(
        args=args_dict, 
        data_input_size=adata.n_vars,
        lr=lr, 
        optimizer=optimizer,
        train_mask_prob_tensor=train_prob_tensor.to(device), 
        val_test_mask_prob_tensor = val_test_prob_tensor.to(device), 
        vis_features_dim=vis_features_dim
        ).to(device)        
        
    print(model.model)

    # Define dict to know whether to maximize or minimize each metric
    max_min_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}

    # Define checkpoint callback to save best model in validation
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        monitor=f'val_{args_dict["optim_metric"]}', # Choose your validation metric
        save_top_k=1, # Save only the best model
        mode=max_min_dict[args_dict["optim_metric"]], # Choose "max" for higher values or "min" for lower values
    )

    # Define the pytorch lightning trainer
    trainer = Trainer(
        max_steps=max_steps,
        val_check_interval=args_dict["val_check_interval"],
        log_every_n_steps=args_dict["val_check_interval"],
        check_val_every_n_epoch=None,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    if train:
        # Train the model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        # Load the best model after training
        best_model_path = checkpoint_callback.best_model_path

    else:
        # Load the checkpoints that will be tested
        best_model_path = load_ckpt_path
        get_performance = True

    if get_performance:
        print(f"Calculating test metrics for model in {load_ckpt_path}")
        # Get performance metrics 
        complete_imputation = get_complete_imputation_results(
                    model = model, 
                    trainer = trainer, 
                    best_model_path = best_model_path, 
                    args = args_dict,
                    prob_tensor = val_test_prob_tensor, 
                    device = device, 
                    prediction_layer=prediction_layer,
                    train_split = train_split, 
                    val_split = val_split, 
                    test_split = test_split
                    )
        
        # complete_imput_results is a dictionary of dictionaries. Each inner dictionary contains the evaluation metrics for the model when imputing randomly masked values in one of the data splits
        complete_imput_results, train_split, val_split, test_split = complete_imputation

        # Save results in a txt file
        test_description = f"Gene imputation through SpaCKLE.\n"\
                            f"Checkpoints restored from {best_model_path}"
        
        file_path = os.path.join(save_path, 'testing_results.txt')
        with open(file_path, 'w') as txt_file:
            txt_file.write(test_description)
            # Convert the dictionary to a formatted string
            dict_str = json.dumps(complete_imput_results, indent=4)
            txt_file.write(dict_str)

        print(f"Results from testing the model in {best_model_path} were saved in {file_path}")
  
# Run gene imputation model
"""
if __name__=='__main__':
    main()
    breakpoint()
"""