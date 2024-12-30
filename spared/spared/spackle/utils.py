import numpy as np
import pandas as pd
import os
import squidpy as sq
import torch
from tqdm import tqdm
import argparse
import anndata as ad
import glob
import pathlib
import sys
#from metrics import get_metrics
from sklearn.preprocessing import StandardScaler

# Path to SpaRED
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Add spared path for imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from metrics.metrics import get_metrics
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

### Function to build or complete dictionary of basic arguments for spackle to work
def get_args_dict(args_dict: dict = None):
    """
    This function prepares the dictionary that contains the arguments necessary for using SpaCKLE. If the user inputs the arguments dictionary,
    this function will check that the dictionary has all the mandatory keys. The values in the default dictionary correspond to those used as default
    when training SpaCKLE in the SpaRED research paper, and the source code contains the description of each key and help to determine new values if needed.

    Args:
        args_dict (dict): A dictionary with the values needed for processing the data and building the model's architecture. 

    Return:
        args_dict (dict): The dictionary with the default arguments values that were missing in the initial args_dict.
    """
    default_args = {
            # Imputation model training and test parameters
            "momentum": 0.9,                         # help = Momentum to use in the optimizer
            "optim_metric": "MSE",                   # help = Metric that should be optimized during training., choices=['PCC-Gene', 'MSE', 'MAE', 'Global']
            "val_check_interval": 10,                # help = Number of steps before doing valid checks.
            "hex_geometry": True,                    # help = Whether the geometry of the spots in the dataset is hexagonal or not.

            # Data loading and handling parameters
            "batch_size": 256,                       # help = The batch size to train model.
            "num_workers": 0,                        # help = DataLoader num_workers parameter - amount of subprocesses to use for data loading.
            "shuffle": True,                         # help = Whether or not to shuffle the data in dataloaders.
            "masking_method": "mask_prob",           # help = The masking method to use., choices=['prob_median', 'mask_prob', 'scale_factor']
            "mask_prob": 0.3,                        # help = float with the probability of masking a gene for imputation when mas_prob masking methos is selected.
            "scale_factor": 0.8,                     # help = The scale factor to use for masking when scale_factor masking method is selected.
            "num_neighs": 18,                        # help = Amount of neighbors to consider for context during imputation.
            
            # SpaCKLE architecture parameters
            "transformer_heads": 1,                 # help = The number of heads in the multiheadattention models of the transformer.
            "transformer_encoder_layers": 2,        # help = The number of sub-encoder-layers in the encoder of the transformer.
            # *** WARNING: The current version of the library does not allow using visual features for gene imputation, thus, the following parameters are ignored during the use of spackle_cleaner in this version. ***
            "use_visual_features": False,           # help = Whether or not to use visual features to guide the imputation process. WARNING: The current version of the library does not allow using visual features for gene imputation.
            "img_backbone": "ViT",                  # help = Backbone to use for image encoding.', choices=['resnet', 'ConvNeXt', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResNet', 'densenet', 'swin']. WARNING: The current version of the library does not allow using visual features for gene imputation.
            "include_genes": True,                  # help = Whether or not to to include the gene expression matrix in the data inputed to the transformer encoder when using visual features. WARNING: The current version of the library does not allow using visual features for gene imputation.
            }
    
    if not args_dict:
        args_dict = default_args

    else:
        args_dict = {key.lower(): value for key, value in args_dict.items()}
        # Add arguments from the default dictionary to args_dict if they are not in it yet.
        for key, value in default_args.items():
            if key not in args_dict:
                args_dict[key] = value
    
    return args_dict

### Define function to get spatial neighbors in an AnnData object
def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
    """
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside a n_hops neighborhood of the observation.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): the size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                used to compute the spatial neighbors and only true for visium datasets.

    Returns:
        dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
    """
    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
    else:
        sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid STNet dataset case

    # Get the adjacency matrix
    adj_matrix = adata.obsp['spatial_connectivities']

    # Define power matrix
    power_matrix = adj_matrix.copy()
    # Define the output matrix
    output_matrix = adj_matrix.copy()

    # Iterate through the hops
    for i in range(n_hops-1):
        # Compute the next hop
        power_matrix = power_matrix * adj_matrix
        # Add the next hop to the output matrix
        output_matrix = output_matrix + power_matrix

    # Zero out the diagonal
    output_matrix.setdiag(0)
    # Threshold the matrix to 0 and 1
    output_matrix = output_matrix.astype(bool).astype(int)

    # Define neighbors dict
    neighbors_dict_index = {}

    # Iterate through the rows of the output matrix
    for i in range(output_matrix.shape[0]):
        # Get the non-zero elements of the row
        non_zero_elements = output_matrix[i].nonzero()[1]
        # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
        neighbors_dict_index[i] = [i] + list(non_zero_elements)
    
    # Return the neighbors dict
    return neighbors_dict_index

### Define cleaning function for single slide:
def adaptive_median_filter_pepper(adata: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool) -> ad.AnnData:
    """
    This function computes the adaptive median filter for pairs (obs, gene) with a zero value (peper noise) in the layer 'from_layer' and
    stores the result in the layer 'to_layer'. The max window size is a neighborhood of n_hops defined by the conectivity hex_geometry
    inputed by parameter. This means the number of concentric rings in a graph to take into account to compute the median.

    Args:
        adata (ad.AnnData): the AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        from_layer (str): the layer to compute the adaptive median filter from.
        to_layer (str): the layer to store the results of the adaptive median filter.
        n_hops (int): the maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                            used to compute the spatial neighbors and only true for visium datasets.

    Returns:
        ad.AnnData: The AnnData object with the results of the adaptive median filter stored in the layer 'to_layer'.
    """
    # Define original expression matrix
    original_exp = adata.layers[from_layer]    

    medians = np.zeros((adata.n_obs, n_hops, adata.n_vars))

    # Iterate over the hops
    for i in range(1, n_hops+1):
        
        # Get dictionary of neighbors for a given number of hops
        curr_neighbors_dict = get_spatial_neighbors(adata, i, hex_geometry)

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

def clean_noise(collection: ad.AnnData, from_layer: str, to_layer: str, n_hops: int, hex_geometry: bool, split_name: str = 'complete') -> ad.AnnData:
    """
    This wrapper function computes the adaptive median filter for all the slides in the collection and then concatenates the results
    into another collection. Details of the adaptive median filter can be found in the adaptive_median_filter_peper function.

    Args:
        collection (ad.AnnData): the AnnData collection to process. Contains all the slides.
        from_layer (str): the layer to compute the adaptive median filter from. Where to clean the noise from.
        to_layer (str): the layer to store the results of the adaptive median filter. Where to store the cleaned data.
        n_hops (int): the maximum number of concentric rings in the graph to take into account to compute the median. Analogous to the max window size.
        hex_geometry (bool): whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                used to compute the spatial neighbors and only true for visium datasets.
        split_name (str, optional): name of the data split being masked and imputed.

    Returns:
        ad.AnnData: The processed AnnData collection with the results of the adaptive median filter stored in the layer 'to_layer'.
    """
    # Print message
    print(f'Applying adaptive median filter to {split_name} collection...')

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

def get_mask_prob_tensor(masking_method, adata, mask_prob=0.3, scale_factor=0.8):
    """
    This function calculates the probability of masking each gene present in the expression matrix. 
    Within this function, there are three different methods for calculating the masking probability, 
    which are differentiated by the 'masking_method' parameter. 
    The return value is a vector of length equal to the number of genes, where each position represents
    the masking probability of that gene.
    
    Args:
        masking_method (str): parameter used to differenciate the method for calculating the probabilities.
        dataset (SpatialDataset): the dataset in a SpatialDataset object.
        mask_prob (float): masking probability for all the genes. Only used when 'masking_method = mask_prob' 
        scale_factor (float): maximum probability of masking a gene if masking_method == 'scale_factor'
    Return:
        prob_tensor (torch.Tensor): vector with the masking probability of each gene for testing. Shape: n_genes  
    """

    # Convert glob_exp_frac to tensor
    glob_exp_frac = torch.tensor(adata.var.glob_exp_frac.values, dtype=torch.float32)
    # Calculate the probability of median imputation
    prob_median = 1 - glob_exp_frac

    if masking_method == "prob_median":
        # Calculate masking probability depending on the prob median
        # (the higher the probability of being replaced with the median, the higher the probability of being masked).
        prob_tensor = prob_median/(1-prob_median)

    elif masking_method == "mask_prob":
        # Calculate masking probability according to mask_prob parameter
        # (Mask everything with the same probability)
        prob_tensor = mask_prob/(1-prob_median)

    elif masking_method == "scale_factor":
        # Calculate masking probability depending on the prob median scaled by a factor
        # (Multiply by a factor the probability of being replaced with median to decrease the masking probability).
        prob_tensor = prob_median/(1-prob_median)
        prob_tensor = prob_tensor*scale_factor
        
    # If probability is more than 1, set it to 1
    prob_tensor[prob_tensor>1] = 1

    return prob_tensor

def mask_exp_matrix(adata: ad.AnnData, pred_layer: str, mask_prob_tensor: torch.Tensor, device):
    """
    This function recieves an adata and masks random values of the pred_layer based on the masking probability of each gene, then saves the masked matrix in the corresponding layer. 
    It also saves the final random_mask for metrics computation. True means the values that are real in the dataset and have been masked for the imputation model development.
    
    Args:
        adata (ad.AnnData): adata of the data split that will be masked and imputed.
        pred_layer (str): indicates the adata.layer with the gene expressions that should be masked and later reconstructed. Shape: spots_in_adata, n_genes
        mask_prob_tensor (torch.Tensor):  tensor with the masking probability of each gene for testing. Shape: n_genes
    
    Return:
        adata (ad.AnnData): adata of the data split with the gene expression matrix already masked and the corresponding random_mask in adata.layers.
    """

    # Extract the expression matrix
    expression_mtx = torch.tensor(adata.layers[pred_layer])
    # Calculate the mask based on probability tensor
    random_mask = torch.rand(expression_mtx.shape).to(device) < mask_prob_tensor.to(device)
    median_imp_mask = torch.tensor(adata.layers['mask']).to(device)
    # Combine random mask with the median imputation mask
    random_mask = random_mask.to(device) & median_imp_mask
    # Mask chosen values.
    expression_mtx[random_mask] = 0
    # Save masked expression matrix in the data_split annData
    adata.layers['masked_expression_matrix'] = np.asarray(expression_mtx.cpu())
    #Save final mask for metric computation
    adata.layers['random_mask'] = np.asarray(random_mask.cpu())

    return adata

# To test the code
if __name__=='__main__':
    hello = 0