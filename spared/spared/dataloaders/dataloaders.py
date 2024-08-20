import anndata as ad
from anndata.experimental.pytorch import AnnLoader
import torch
import os
import glob
import json
from time import time
from datetime import datetime
from torch_geometric.loader import DataLoader as geo_DataLoader
import numpy as np
import pathlib
from typing import Tuple
import sys
from typing import Tuple

# Path a spared
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from layer_operations import layer_operations
from spot_features import spot_features
from graph_operations import graph_operations
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))


# TODO: Fix the internal fixme (DISCUSS AGAIN)
def get_pretrain_dataloaders(adata: ad.AnnData, layer: str = 'c_d_log1p', batch_size: int = 128, shuffle: bool = True, use_cuda: bool = False) -> Tuple[AnnLoader, AnnLoader, AnnLoader]:
    """ Get dataloaders for pretraining an image encoder.
    This function returns the dataloaders for training an image encoder. This means training a purely vision-based model on only
    the patches to predict the gene expression of the patches.

    Dataloaders are returned as a tuple, if there is no test set for the dataset, then the test dataloader is None.

    Args:
        adata (ad.AnnData): The AnnData object that will be processed.
        layer (str, optional): The layer to use for the pre-training. The adata.X will be set to that of 'layer'. Defaults to 'deltas'.
        batch_size (int, optional): The batch size of the loaders. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data in the loaders. Defaults to True.
        use_cuda (bool, optional): True for using cuda in the loader. Defaults to False.

    Returns:
        Tuple[AnnLoader, AnnLoader, AnnLoader]: The train, validation and test dataloaders. If there is no test set, the test dataloader is None.
    """
    # Get the sample indexes for the train, validation and test sets
    idx_train, idx_val, idx_test = adata.obs[adata.obs.split == 'train'].index, adata.obs[adata.obs.split == 'val'].index, adata.obs[adata.obs.split == 'test'].index

    ##### Addition to handle noisy training #####

    # FIXME: Put this in a part of the complete processing pipeline instead of the dataloader function. 
    # Handle noisy training
    # Add this function in procces_data function and automaticaaly generate noisy layers for this layers:
    # c_d_log1p, c_t_log1p, c_d_deltas, c_t_deltas
    # FIXME: This is generating the unwanted message "Using noisy_delta layer for training. This will probably yield bad results." in quickstart tutorial
    adata = layer_operations.add_noisy_layer(adata=adata, prediction_layer=layer)

    # Set the X of the adata to the layer casted to float32
    adata.X = adata.layers[layer].astype(np.float32)

    imp_model_str = 'transformer model' if layer in ['c_t_log1p', 'c_t_deltas'] else 'median filter'

    # Print with the percentage of the dataset that was replaced
    imp_pct = 100 * (~adata.layers["mask"]).sum() / (adata.n_vars*adata.n_obs)
    print('Percentage of imputed observations with {}: {:5.3f}%'.format(imp_model_str, imp_pct))

    # If the prediction layer is some form of deltas, add the used mean of the layer as a column in the var
    if 'deltas' in layer:
        # Add a var column of used means of the layer
        mean_key = f'{layer}_avg_exp'.replace('deltas', 'log1p')
        adata.var['used_mean'] = adata.var[mean_key]

    # Subset the global data handle also the possibility that there is no test set
    adata_train, adata_val = adata[idx_train, :], adata[idx_val, :]
    adata_test = adata[idx_test, :] if len(idx_test) > 0 else None

    # Declare dataloaders
    train_dataloader = AnnLoader(adata_train, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
    val_dataloader = AnnLoader(adata_val, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda)
    test_dataloader = AnnLoader(adata_test, batch_size=batch_size, shuffle=shuffle, use_cuda=use_cuda) if adata_test is not None else None

    return train_dataloader, val_dataloader, test_dataloader

# TODO: Fix the internal fixme (DEPENDS ON THE PREVIOUS DISCUSSION)
def get_graph_dataloaders(adata: ad.AnnData, dataset_path: str='', layer: str = 'c_t_log1p', n_hops: int = 2, backbone: str ='densenet', model_path: str = "None", batch_size: int = 128, 
                          shuffle: bool = True, hex_geometry: bool=True, patch_size: int=224) -> Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]:
    """ Get dataloaders for the graphs of a dataset.
    This function performs all the pipeline to get graphs dataloaders for a dataset. It does the following steps:

        1. Computes embeddings and predictions for the patches using the specified backbone and model.
        2. Computes the graph dictionaries for the dataset using the embeddings and predictions.
        3. Saves the graphs in the dataset_path folder.
        4. Returns the train, validation and test dataloaders for the graphs.
    
    The function also checks if the graphs are already saved in the dataset_path folder. If they are, it loads them instead of recomputing them. In case 
    the dataset has no test set, the test dataloader is set to None.

    Args:
        adata (ad.AnnData): The AnnData object to process.
        dataset_path (str, optional): The path to the dataset (where the graphs will be stored). Defaults to ''.
        layer (str, optional): Layer to predict. Defaults to 'c_t_log1p'.
        n_hops (int, optional): Number of hops to compute the graph. Defaults to 2.
        backbone (str, optional): Backbone model to use. Defaults to 'densenet'.
        model_path (str, optional): Path to the model to use. Defaults to "None".
        batch_size (int, optional): Batch size of the dataloaders. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data in the dataloaders. Defaults to True.
        hex_geometry (bool, optional): Whether the graph is hexagonal or not. Defaults to True.
        patch_size (int, optional): Size of the patches. Defaults to 224.

    Returns:
        Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]: _description_
    """
    # Get dictionary of parameters to get the graphs
    curr_graph_params = {
        'n_hops': n_hops,
        'layer': layer,
        'backbone': backbone,
        'model_path': model_path
    }        

    # Create graph directory if it does not exist
    os.makedirs(os.path.join(dataset_path, 'graphs'), exist_ok=True)
    # Get the filenames of the parameters of all directories in the graph folder
    filenames = glob.glob(os.path.join(dataset_path, 'graphs', '**', 'graph_params.json' ), recursive=True)

    # Define boolean to check if the graphs are already saved
    found_graphs = False

    # Iterate over all the filenames and check if the parameters are the same
    for filename in filenames:
        with open(filename, 'r') as f:
            # Load the parameters of the dataset
            saved_params = json.load(f)
            # Check if the parameters are the same
            if saved_params == curr_graph_params:
                print(f'Graph data already saved in {filename}')
                found_graphs = True
                # Track the time and load the graphs
                start = time()
                train_graphs = torch.load(os.path.join(os.path.dirname(filename), 'train_graphs.pt'))
                val_graphs = torch.load(os.path.join(os.path.dirname(filename), 'val_graphs.pt'))
                test_graphs = torch.load(os.path.join(os.path.dirname(filename), 'test_graphs.pt')) if os.path.exists(os.path.join(os.path.dirname(filename), 'test_graphs.pt')) else None
                print(f'Loaded graphs in {time() - start:.2f} seconds.')
                break

    # If the graphs are not found, compute them
    if not found_graphs:
        
        # Print that we are computing the graphs
        print('Graphs not found in file, computing graphs...')

        # FIXME: Again this should be in the processing part and not in the dataloader
        adata = layer_operations.add_noisy_layer(adata=adata, prediction_layer=layer)

        # We compute the embeddings and predictions for the patches
        spot_features.compute_patches_embeddings(adata=adata, backbone=backbone, model_path=model_path, patch_size=patch_size)
        spot_features.compute_patches_predictions(adata=adata, backbone=backbone, model_path=model_path, patch_size=patch_size)
        
        # Get graph dicts
        general_graph_dict = graph_operations.get_graphs(adata=adata, n_hops=n_hops, layer=layer, hex_geometry=hex_geometry)

        # Get the train, validation and test indexes
        idx_train, idx_val, idx_test = adata.obs[adata.obs.split == 'train'].index, adata.obs[adata.obs.split == 'val'].index, adata.obs[adata.obs.split == 'test'].index

        # Get list of graphs
        train_graphs = [general_graph_dict[idx] for idx in idx_train]
        val_graphs = [general_graph_dict[idx] for idx in idx_val]
        test_graphs = [general_graph_dict[idx] for idx in idx_test] if len(idx_test) > 0 else None

        print('Saving graphs...')
        # Create graph directory if it does not exist with the current time
        graph_dir = os.path.join(dataset_path, 'graphs', datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        os.makedirs(graph_dir, exist_ok=True)

        # Save the graph parameters
        with open(os.path.join(graph_dir, 'graph_params.json'), 'w') as f:
            # Write the json
            json.dump(curr_graph_params, f, indent=4)

        torch.save(train_graphs, os.path.join(graph_dir, 'train_graphs.pt'))
        torch.save(val_graphs, os.path.join(graph_dir, 'val_graphs.pt'))
        torch.save(test_graphs, os.path.join(graph_dir, 'test_graphs.pt')) if test_graphs is not None else None
    

    # Declare dataloaders
    train_dataloader = geo_DataLoader(train_graphs, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = geo_DataLoader(val_graphs, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = geo_DataLoader(test_graphs, batch_size=batch_size, shuffle=shuffle) if test_graphs is not None else None

    return train_dataloader, val_dataloader, test_dataloader