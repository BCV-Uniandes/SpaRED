import anndata as ad
from tqdm import tqdm
import numpy as np
import sys
import pathlib
from datetime import datetime
import torch
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import json
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
import warnings
# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from spot_features import spot_features
from layer_operations import layer_operations
from datasets import datasets
from spackle.utils import *
from spackle.model import GeneImputationModel
from spackle.dataset import ImputationDataset
from spackle.main import train_spackle
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))


#clean noise with medians
# TODO: Think in making this function also add the binary mask layer
def median_cleaner(collection: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
    """Remove noise with adaptive median filter.

    Function that cleans noise (missing data) with the modified adaptive median initially proposed by `SEPAL <https://doi.org/10.48550/arXiv.2309.01036>`_
    filter for each slide in an AnnData collection. Windows to compute the medians are defined by topological distances (hops) in the neighbors graph defined
    by the ``hex_geometry`` parameter with a maximum window size of ``n_hops``. The adaptive median filter denoises each gene independently. In other words
    gene A has no influence on the denoising of gene B. The data will be taken from ``adata.layers[from_layer]`` and the results will be stored in
    ``adata.layers[to_layer]``.

    Args:
        collection (ad.AnnData): The AnnData collection to process.
        from_layer (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
        to_layer (str): The layer to store the results of the adaptive median filter. Where to store the cleaned data.
        n_hops (int): The maximum number of concentric rings in the neighbors graph to take into account to compute the median. Analogous to the maximum window size.
        hex_geometry (bool): ``True`` if the graph has hexagonal spatial geometry (Visium technology). If ``False``, then the graph is a grid.

    Returns:
        ad.AnnData: New AnnData collection with the results of the adaptive median filter stored in ``adata.layers[to_layer]``.
    """

    ### Define cleaning function for single slide:
    def adaptive_median_filter_pepper(adata: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
        """
        This function computes a modified adaptive median filter for pairs (obs, gene) with a zero value (peper noise) in the layer 'from_layer' and
        stores the result in the layer 'to_layer'. The max window size is a neighborhood of n_hops defined by the conectivity (hexagonal or grid).
        This means the number of concentric rings in a graph to take into account to compute the median.

        The adaptive median filter denoises each gene independently. In other words gene A has no influence on the denoising of gene B.

        Args:
            adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
            from_layer (str): The layer to compute the adaptive median filter from.
            to_layer (str): The layer to store the results of the adaptive median filter.
            n_hops (int): The maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
            hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                 true for visium datasets.

        Returns:
            ad.AnnData: The AnnData object with the results of the adaptive median filter stored in the layer 'to_layer'.
        """
        # Define original expression matrix
        original_exp = adata.layers[from_layer]    

        medians = np.zeros((adata.n_obs, n_hops, adata.n_vars))

        # Iterate over the hops
        for i in range(1, n_hops+1):
            
            # Get dictionary of neighbors for a given number of hops
            curr_neighbors_dict = spot_features.get_spatial_neighbors(adata, i, hex_geometry)

            # Iterate over observations
            for j in range(adata.n_obs):
                # Get the list of indexes of the neighbors of the j'th observation
                neighbors_idx = curr_neighbors_dict[j]
                # Get the expression matrix of the neighbors
                neighbor_exp = original_exp[neighbors_idx, :]
                # Get the median of the expression matrix
                median = np.median(neighbor_exp, axis=0)

                # Store the median in the medians matrix
                medians[j, i-1, :] = median
        
        # Also robustly compute the median of the non-zero values for each gene
        general_medians = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0, original_exp)
        general_medians[np.isnan(general_medians)] = 0.0 # Correct for possible nans

        # Define corrected expression matrix
        corrected_exp = np.zeros_like(original_exp)

        ### Now that all the possible medians are computed. We code for each observation:
        
        # Note: i indexes over observations, j indexes over genes
        for i in range(adata.n_obs):
            for j in range(adata.n_vars):
                
                # Get real expression value
                z_xy = original_exp[i, j]

                # Only apply adaptive median filter if real expression is zero
                if z_xy != 0:
                    corrected_exp[i,j] = z_xy
                    continue
                
                else:

                    # Definie initial stage and window size
                    current_stage = 'A'
                    k = 0

                    while True:

                        # Stage A:
                        if current_stage == 'A':
                            
                            # Get median value
                            z_med = medians[i, k, j]

                            # If median is not zero then go to stage B
                            if z_med != 0:
                                current_stage = 'B'
                                continue
                            # If median is zero, then increase window and repeat stage A
                            else:
                                k += 1
                                if k < n_hops:
                                    current_stage = 'A'
                                    continue
                                # If we have the biggest window size, then return the median
                                else:
                                    # NOTE: Big modification to the median filter here. Be careful
                                    corrected_exp[i,j] = general_medians[j]
                                    break


                        # Stage B:
                        elif current_stage == 'B':
                            
                            # Get window median
                            z_med = medians[i, k, j]

                            # If real expression is not peper then return it
                            if z_xy != 0:
                                corrected_exp[i,j] = z_xy
                                break
                            # If real expression is peper, then return the median
                            else:
                                corrected_exp[i,j] = z_med
                                break

        # Add corrected expression to adata
        adata.layers[to_layer] = corrected_exp

        return adata

    # Print message
    print('Applying adaptive median filter to collection...')

    # Get the unique slides
    slides = np.unique(collection.obs['slide_id'])

    # Define the corrected adata list
    corrected_adata_list = []

    # Iterate over the slides
    for slide in tqdm(slides):
        # Get the adata of the slide
        adata = collection[collection.obs['slide_id'] == slide].copy()
        # Apply adaptive median filter
        adata = adaptive_median_filter_pepper(adata, from_layer, to_layer, n_hops, hex_geometry)
        # Append to the corrected adata list
        corrected_adata_list.append(adata)
    
    # Concatenate the corrected adata list
    corrected_collection = ad.concat(corrected_adata_list, join='inner', merge='same')
    # Restore the uns attribute
    corrected_collection.uns = collection.uns

    return corrected_collection

#Replicate SpaCKLE's results
def spackle_cleaner_experiment(adata: ad.AnnData, dataset: str, from_layer: str, device, args_dict = None, lr = 1e-3, train = True, load_ckpt_path = "", optimizer = "Adam", max_steps = 1000) -> ad.AnnData:
    """This function's purpose is solely to reproduce the results presented in SpaCKLE's paper.

    Function that cleans noise (completes missing data) with a SpaCKLE model that can be either trained or loaded as a pre-trained model from the original published checkpoints.
    The data will be taken from ``adata.layers[from_layer]`` and the results will be stored in ``adata.layers[to_layer]``. If training a new SpaCKLE
    model, it will be saved in the path ``imput_results/[dataset_name]/[run_date]``.

    Args:
        adata (ad.AnnData): The AnnData collection to process. The adata must have pre-determined data splits in ``adata.obs['split']`` and the values should be ``train``, ``val``, and (optional) ``test``.
        dataset (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
        from_layer (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
        to_layer (str): The layer to store the results of the adaptive median filter. Where to store the cleaned data.
        device (torch.device): device in which tensors will be processed.
        args_dict (dict): A dictionary with the values needed for processing the data and building the model's architecture. For more information on the required keys, refer to the 
                          documentation of the function ``get_args_dict()`` in `spared.spackle.utils`.
        lr (float): The learning rate for training the model.
        train (bool): If True, a new SpaCKLE model will be trained and tested, otherwise, the function will only test the pretrained model found in ``load_ckpt_path``.
        get_performance (bool): If True, the function will calculate the final evaluation metrics of the model and save them in a txt file in save_path.
        load_ckpt_path (str): Path to the checkpoints of a pretrained SpaCKLE model. This path should lead directly to the .ckpt file.
        optimizer (str, optional): The name of the optimizer selected for the training process. Default = "Adam".
        max_steps (int, optional): Stop training after this number of steps. Default = 1000.

    Returns:
        adata (ad.AnnData): The input AnnData collection with the added cleaned layer in ``adata.layers[to_layer]``.
        load_ckpt_path (str): Path to the checkpoints of the trained SpaCKLE model.
    """
    
    # Get datetime
    run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Set manual seeds and get cuda
    seed_everything(42)

    # Check args_dict and fill missing values or create args dictionary in case user does not input it
    args_dict = get_args_dict(args_dict)
    
    # TODO: [PC] allow the use of an already-trained model?
    if train:
        # Create directory where the newly trained model will be saved #save_path = os.path.join('imput_results', dataset, "best_model") # TODO: [PC] group opinion: ¿should we set data naming with date like in our works?
        save_path = os.path.join('imput_results', dataset, run_date)
        os.makedirs(save_path, exist_ok=True)

        # Save script arguments in json file
        with open(os.path.join(save_path, 'script_params.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)

        print(f"Training a new SpaCKLE model. The script arguments and best checkpoints will be saved in {save_path}")

    else:

        assert os.path.exists(load_ckpt_path), "load_ckpts_path not found. Please use train = True if you do not have the checkpoints of a trained SpaCKLE model and its corresponding script_params.json file."
        
        save_path = os.path.dirname(load_ckpt_path)
        # FIXME: [PC] decidir qué elementos comparar y recordar que al subir los pesos (i.e Drive) subirlos con su json de params correspondiente
        with open(os.path.join(save_path, 'script_params.json'), 'r') as f:
            saved_script_params = json.load(f)
            # Check that the parameters of the loaded model agree with the current inference process
            #if (saved_script_params['prediction_layer'] != args_dict['prediction_layer']) or (saved_script_params['prediction_layer'] != args_dict['prediction_layer']):
            #    warnings.warn("Saved model's parameters differ from those of the current argparse.")

        print(f"Model from {load_ckpt_path} will be loaded and tested. No new training will be undergone.")
        
    # Train new SpaCKLE model
    train_spackle(
        adata=adata, 
        device=device, 
        save_path=save_path, 
        prediction_layer=from_layer, 
        lr=lr, 
        train=train, 
        get_performance=True,
        load_ckpt_path=load_ckpt_path, 
        optimizer=optimizer, 
        max_steps=max_steps, 
        args_dict=args_dict)

#clean noise con spackle
def spackle_cleaner(adata: ad.AnnData, dataset: str, from_layer: str, to_layer: str, device, args_dict = None, lr = 1e-3, train = True, get_performance_metrics = True, load_ckpt_path = "", optimizer = "Adam", max_steps = 1000) -> ad.AnnData:
    """Remove noise with SpaCKLE.

    Function that cleans noise (completes missing data) with a SpaCKLE model that can be either trained or loaded as a pre-trained model.
    The data will be taken from ``adata.layers[from_layer]`` and the results will be stored in ``adata.layers[to_layer]``. If training a new SpaCKLE
    model, it will be saved in the path ``imput_results/[dataset_name]/[run_date]``.

    Args:
        adata (ad.AnnData): The AnnData collection to process. The adata must have pre-determined data splits in ``adata.obs['split']`` and the values should be ``train``, ``val``, and (optional) ``test``.
        dataset (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
        from_layer (str): The layer to compute the adaptive median filter from. Where to clean the noise from.
        to_layer (str): The layer to store the results of the adaptive median filter. Where to store the cleaned data.
        device (torch.device): device in which tensors will be processed.
        args_dict (dict): A dictionary with the values needed for processing the data and building the model's architecture. For more information on the required keys, refer to the 
                          documentation of the function ``get_args_dict()`` in `spared.spackle.utils`.
        lr (float): The learning rate for training the model.
        train (bool): If True, a new SpaCKLE model will be trained and tested, otherwise, the function will only test the pretrained model found in ``load_ckpt_path``.
        get_performance (bool): If True, the function will calculate the final evaluation metrics of the model and save them in a txt file in save_path.
        load_ckpt_path (str): Path to the checkpoints of a pretrained SpaCKLE model. This path should lead directly to the .ckpt file.
        optimizer (str, optional): The name of the optimizer selected for the training process. Default = "Adam".
        max_steps (int, optional): Stop training after this number of steps. Default = 1000.

    Returns:
        adata (ad.AnnData): The input AnnData collection with the added cleaned layer in ``adata.layers[to_layer]``.
        load_ckpt_path (str): Path to the checkpoints of the trained SpaCKLE model.
    """
    
    # Get datetime
    run_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Set manual seeds and get cuda
    seed_everything(42)

    # Check args_dict and fill missing values or create args dictionary in case user does not input it
    args_dict = get_args_dict(args_dict)
    
    if train:
        # Código para entrenar modelo (train_splackle()) y retornar ruta a mejores pesos del entrenamiento
        # Create directory where the newly trained model will be saved 
        save_path = os.path.join('imput_results', dataset, run_date)
        os.makedirs(save_path, exist_ok=True)

        # Save script arguments in json file
        with open(os.path.join(save_path, 'script_params.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)
        
        # Train new SpaCKLE model
        train_spackle(
            adata=adata, 
            device=device, 
            save_path=save_path, 
            prediction_layer=from_layer, 
            lr=lr, 
            train=train, 
            get_performance=get_performance_metrics,
            load_ckpt_path=load_ckpt_path, 
            optimizer=optimizer, 
            max_steps=max_steps, 
            args_dict=args_dict)
        
        load_ckpt_path = glob.glob(os.path.join(save_path, '*.ckpt'))[0]

    else:
        assert os.path.exists(load_ckpt_path), "load_ckpts_path not found. Please use train = True if you do not have the checkpoints of a trained SpaCKLE model and its corresponding script_params.json file."
        
        save_path = os.path.dirname(load_ckpt_path)
        # FIXME: [PC] decidir qué elementos comparar y recordar que al subir los pesos (i.e Drive) subirlos con su json de params correspondiente
        with open(os.path.join(save_path, 'script_params.json'), 'r') as f:
            saved_script_params = json.load(f)
            # Check that the parameters of the loaded model agree with the current inference process
            #if (saved_script_params['prediction_layer'] != args_dict['prediction_layer']) or (saved_script_params['prediction_layer'] != args_dict['prediction_layer']):
            #    warnings.warn("Saved model's parameters differ from those of the current argparse.")

            if saved_script_params['transformer_dim'] != adata.n_vars:
                warnings.warn("The architecture of the model you want to load may not be compatible with the shape of the data.")

    ## Run SpaCKLE model to complete gene data that is missing in adata either with a recently trained model or using the checkpoints of a pretrained model, depending on the parameters selected.
    # Declare model
    vis_features_dim = 0
    model = GeneImputationModel(
        args=args_dict, 
        data_input_size=adata.n_vars,
        lr=lr,
        optimizer=optimizer,
        vis_features_dim=vis_features_dim
        ).to(device)  

    # Load best checkpoints
    state_dict = torch.load(load_ckpt_path)
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Finished loading model with weights from {load_ckpt_path}")

    # Prepare data and dataloader
    data = ImputationDataset(adata, args_dict, 'complete', from_layer)
    dataloader = DataLoader(
        data, 
        batch_size=args_dict['batch_size'], 
        shuffle=False, 
        pin_memory=True, 
        drop_last=False, 
        num_workers=args_dict['num_workers'])
    
    # Get gene imputations for missing values of randomly masked elements trhoughout the entire dataset
    all_exps = []
    all_masks = []
    exp_with_imputation = []
    
    print("----"*30)
    print(f"Completing missing values in adata with the SpaCKLE model from {load_ckpt_path}")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            del batch['split_name']
            # Extract batch variables
            batch = {k: v.to(device) for k, v in batch.items()}
            expression_gt = batch['exp_matrix_gt']
            mask = batch['real_missing']
            # Remove median imputations from gene expression matrix
            input_genes = expression_gt.clone()
            input_genes[~mask] = 0

            # Get predictions
            prediction = model.forward(input_genes)

            # Imput predicted gene expression only in missing data for 'main spot' in the neighborhood
            imputed_exp = torch.where(mask[:,0,:], expression_gt[:,0,:], prediction[:,0,:])

            all_exps.append(expression_gt[:,0,:])
            all_masks.append(batch['real_missing'][:,0,:])
            exp_with_imputation.append(imputed_exp)

    # Concatenate output tensors into complete data expression matrix
    all_exps = torch.cat(all_exps)
    all_masks = torch.cat(all_masks)
    exp_with_imputation = torch.cat(exp_with_imputation) 

    # Add imputed data to adata
    adata.layers[to_layer] = np.asarray(exp_with_imputation.cpu().double())
    
    # Return the adata with cleaned layer and the path to the ckpts used to complete the missing values.
    return adata, load_ckpt_path