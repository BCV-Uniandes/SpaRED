import pathlib
import sys
import anndata as ad
import os
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import plotting
import spot_features
import models
import dataloaders

#get dataset
data = datasets.get_dataset("vicari_mouse_brain", visualize=False)

# Declare train and test loaders
train_dataloader, val_dataloader, test_dataloader = dataloaders.get_pretrain_dataloaders(
    adata=data.adata,
    layer = 'c_t_log1p',
    batch_size = 265,
    shuffle = True,
    use_cuda = True
)

# Define argparse variables
test_args = argparse.Namespace()
arg_dict = vars(test_args)
input_dict = {
    'img_backbone': 'ShuffleNetV2',
    'img_use_pretrained': True,
    'average_test': False,
    'optim_metric': 'MSE',
    'robust_loss': False,
    'optimizer': 'Adam',
    'lr': 0.0001,
    'momentum': 0.9,
}

for key,value in input_dict.items():
    arg_dict[key]= value


# Declare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.ImageBackbone(args=test_args,  latent_dim=data.adata.n_vars).to(device)

# Define checkpoint callback to save best model in validation
checkpoint_callback = ModelCheckpoint(
    monitor=f'val_MSE', # Choose your validation metric
    save_top_k=1, # Save only the best model
    mode='min'
)

# Define the trainier and fit the model
trainer = Trainer(
    max_steps=30,
    val_check_interval=10,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=None,
    devices=1,
    enable_progress_bar=True,
    enable_model_summary=True
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# Load the best model after training
best_model_path = checkpoint_callback.best_model_path
print(best_model_path)
model = models.ImageBackbone.load_from_checkpoint(best_model_path)

# Test model if there is a test dataloader
if not (test_dataloader is None):
    trainer.test(model, dataloaders=test_dataloader)

#best_model_path = "/media/SSD4/dvegaa/SpaRED/spared/tutorials/lightning_logs/version_5/checkpoints/epoch=0-step=30.ckpt"
#model = models.ImageBackbone.load_from_checkpoint(best_model_path)

def get_predictions(dataloader, model, device="cuda")->None:
    # Set model to eval mode
    model=model.to(device)
    model.eval()

    glob_expression_pred = None
    # Get complete predictions
    with torch.no_grad():
        for b in tqdm(range(0,len(test_dataloader.dataset))):
            batch = dataloader.dataset[b]
            expression_pred, expression_gt, mask = model.test_step(batch)
            expression_pred = expression_pred.cpu()

            # Concat batch to get global predictions and IDs
            glob_expression_pred = expression_pred if glob_expression_pred is None else torch.cat((glob_expression_pred, expression_pred))

    return glob_expression_pred

glob_expression_pred = get_predictions(test_dataloader, model)

breakpoint()
# Put complete predictions in a single dataframe
adata_test = data.adata[data.adata.obs["split"] == "test"]
pred_matrix = glob_expression_pred
glob_ids = adata_test.obs['unique_id'].tolist()
pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=adata_test.var_names)
pred_df = pred_df.reindex(adata_test.obs.index)

# Add layer to adata
layer = "c_t_log1p"
adata_test.layers[f'predictions,{layer}'] = pred_df

os.makedirs('./prediction_images', exist_ok=True)
plotting.log_pred_image(adata_test, n_genes=2, slides={}, save_path='./prediction_images')