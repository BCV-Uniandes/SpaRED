import anndata as ad
import torch
import torchvision.models as tmodels
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import scanpy as sc
import squidpy as sq
import sys
import pathlib

# Path a spared 
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import models.py file
from models import models
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

### Patch processing functions
## Compute the patch embeddings
def compute_patches_embeddings(adata: ad.AnnData, backbone: str ='densenet', model_path:str="None", patch_size: int = 224) -> None:
    """ Compute embeddings or predictions for patches.

    This function computes embeddings for a given backbone model and adata object. It can optionally
    compute using a stored model in model_path or a pretrained model from pytorch. The embeddings are
    stored in adata.obsm[f'embeddings_{backbone}']. The patches
    must be stored in adata.obsm[f'patches_scale_{patch_scale}'] and must be of shape (n_patches, patch_size*patch_size*3).

    The function only modifies the AnnData object in place.

    Args:
        adata (ad.AnnData): The AnnData object to process.
        backbone (str, optional): A string specifiying the backbone model to use. Must be one of the following ['resnet', 'resnet50', 'ConvNeXt', 'EfficientNetV2', 'InceptionV3', 'MaxVit', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResnet', 'densenet', 'swin']. Defaults to 'densenet'.
        model_path (str, optional): The path to a stored model. If set to 'None', then a pretrained model is used. Defaults to "None".
        patch_size (int, optional): The size of the patches. Defaults to 224.

    Raises:
        ValueError: If the backbone is not supported.
    """
    # Define a cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verify that the patch scale exists and only exists once
    obsm_keys = list(adata.obsm.keys())
    patch_scale_key = [key for key in obsm_keys if "patches_scale" in key]
    assert len(patch_scale_key) == 1, "patches_scale key either does not exist or exists more than once in keys_list."

    # Get the patch scale
    patch_scale = patch_scale_key[0].split('_')[-1]
    
    # Define the model
    model = models.ImageEncoder(backbone=backbone, use_pretrained=True, latent_dim=adata.n_vars)

    if model_path != "None":
        saved_model = torch.load(model_path)
        # Check if state_dict is inside a nested dictionary
        if 'state_dict' in saved_model.keys():
            saved_model = saved_model['state_dict']

        model.load_state_dict(saved_model)
    
    # Define the weights for the model depending on the backbone
    if backbone == 'resnet':
        weights = tmodels.ResNet18_Weights.DEFAULT
        model.encoder.fc = nn.Identity()
    elif backbone == 'resnet50':
        weights = tmodels.ResNet50_Weights.DEFAULT
        model.encoder.fc = nn.Identity()
    elif backbone == 'ConvNeXt':
        weights = tmodels.ConvNeXt_Tiny_Weights.DEFAULT
        model.encoder.classifier[2] = nn.Identity()
    elif backbone == 'EfficientNetV2':
        weights = tmodels.EfficientNet_V2_S_Weights.DEFAULT 
        model.encoder.classifier[1] = nn.Identity()
    elif backbone == 'InceptionV3':
        weights = tmodels.Inception_V3_Weights.DEFAULT
        model.encoder.fc = nn.Identity()
    elif backbone == "MaxVit":
        weights = tmodels.MaxVit_T_Weights.DEFAULT
        model.encoder.classifier[5] = nn.Identity()
    elif backbone == "MobileNetV3":
        weights = tmodels.MobileNet_V3_Small_Weights.DEFAULT
        model.encoder.classifier[3] = nn.Identity()
    elif backbone == "ResNetXt":
        weights = tmodels.ResNeXt50_32X4D_Weights.DEFAULT
        model.encoder.fc = nn.Identity()
    elif backbone == "ShuffleNetV2":
        weights = tmodels.ShuffleNet_V2_X0_5_Weights.DEFAULT
        model.encoder.fc = nn.Identity()
    elif backbone == "ViT":
        weights = tmodels.ViT_B_16_Weights.DEFAULT
        model.encoder.heads.head = nn.Identity()
    elif backbone == "WideResnet":
        weights = tmodels.Wide_ResNet50_2_Weights.DEFAULT
        model.encoder.fc = nn.Identity()
    elif backbone == 'densenet':
        weights = tmodels.DenseNet121_Weights.DEFAULT
        model.encoder.classifier = nn.Identity() 
    elif backbone == 'swin':
        weights = tmodels.Swin_T_Weights.DEFAULT
        model.encoder.head = nn.Identity()
    else:
        raise ValueError(f'Backbone {backbone} not supported')

    # Pass model to device and put in eval mode
    model.to(device)
    model.eval()

    # Perform specific preprocessing for the model
    preprocess = weights.transforms()

    # Get the patches
    # patch_scale = 1.0
    flat_patches = adata.obsm[f'patches_scale_{patch_scale}']

    # Reshape all the patches to the original shape
    all_patches = flat_patches.reshape((-1, patch_size, patch_size, 3))
    torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
    rescaled_patches = torch_patches / 255                                       # Rescale patches to [0, 1]
    processed_patches = preprocess(rescaled_patches)                             # Preprocess patches
    
    # Create a dataloader
    dataloader = DataLoader(processed_patches, batch_size=256, shuffle=False, num_workers=4)

    # Declare lists to store the embeddings 
    outputs = []

    with torch.no_grad():
        desc = 'Getting embeddings'
        
        for batch in tqdm(dataloader, desc=desc):
            batch = batch.to(device)                    # Send batch to device                
            batch_output = model(batch)                 # Get embeddings 
            outputs.append(batch_output)                # Append to list


    # Concatenate all embeddings
    outputs = torch.cat(outputs, dim=0)

    # Pass embeddingsto cpu and add to data.obsm
    adata.obsm[f'embeddings_{backbone}'] = outputs.cpu().numpy()

## Compute the patch predictions
def compute_patches_predictions(adata: ad.AnnData, backbone: str ='densenet', model_path:str="None", patch_size: int = 224) -> None:
    """ Compute predictions for patches.

    This function computes predictions for a given backbone model and adata object. It can optionally
    compute using a stored model in model_path or a pretrained model from pytorch. The predictions are
    stored in adata.obsm[f'predictions_{backbone}']. The patches
    must be stored in adata.obsm[f'patches_scale_{patch_scale}'] and must be of shape (n_patches, patch_size*patch_size*3).

    The function only modifies the AnnData object in place.

    Args:
        adata (ad.AnnData): The AnnData object to process.
        backbone (str, optional): A string specifiying the backbone model to use. Must be one of the following ['resnet', 'resnet50', 'ConvNeXt', 'EfficientNetV2', 'InceptionV3', 'MaxVit', 'MobileNetV3', 'ResNetXt', 'ShuffleNetV2', 'ViT', 'WideResnet', 'densenet', 'swin']. Defaults to 'densenet'.
        model_path (str, optional): The path to a stored model. If set to 'None', then a pretrained model is used. Defaults to "None".
        patch_size (int, optional): The size of the patches. Defaults to 224.

    Raises:
        ValueError: If the backbone is not supported.
    """

    # Define a cuda device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verify that the patch scale exists and only exists once
    obsm_keys = list(adata.obsm.keys())
    patch_scale_key = [key for key in obsm_keys if "patches_scale" in key]
    assert len(patch_scale_key) == 1, "patches_scale key either does not exist or exists more than once in keys_list."

    # Get the patch scale
    patch_scale = patch_scale_key[0].split('_')[-1]
    
    # Define the model
    model = models.ImageEncoder(backbone=backbone, use_pretrained=True, latent_dim=adata.n_vars)

    if model_path != "None":
        saved_model = torch.load(model_path)
        # Check if state_dict is inside a nested dictionary
        if 'state_dict' in saved_model.keys():
            saved_model = saved_model['state_dict']

        model.load_state_dict(saved_model)
    
    # Define the weights for the model depending on the backbone
    if backbone == 'resnet':
        weights = tmodels.ResNet18_Weights.DEFAULT
    elif backbone == 'resnet50':
        weights = tmodels.ResNet50_Weights.DEFAULT
    elif backbone == 'ConvNeXt':
        weights = tmodels.ConvNeXt_Tiny_Weights.DEFAULT
    elif backbone == 'EfficientNetV2':
        weights = tmodels.EfficientNet_V2_S_Weights.DEFAULT 
    elif backbone == 'InceptionV3':
        weights = tmodels.Inception_V3_Weights.DEFAULT
    elif backbone == "MaxVit":
        weights = tmodels.MaxVit_T_Weights.DEFAULT
    elif backbone == "MobileNetV3":
        weights = tmodels.MobileNet_V3_Small_Weights.DEFAULT
    elif backbone == "ResNetXt":
        weights = tmodels.ResNeXt50_32X4D_Weights.DEFAULT
    elif backbone == "ShuffleNetV2":
        weights = tmodels.ShuffleNet_V2_X0_5_Weights.DEFAULT
    elif backbone == "ViT":
        weights = tmodels.ViT_B_16_Weights.DEFAULT
    elif backbone == "WideResnet":
        weights = tmodels.Wide_ResNet50_2_Weights.DEFAULT
    elif backbone == 'densenet':
        weights = tmodels.DenseNet121_Weights.DEFAULT
    elif backbone == 'swin':
        weights = tmodels.Swin_T_Weights.DEFAULT
    else:
        raise ValueError(f'Backbone {backbone} not supported')

    # Pass model to device and put in eval mode
    model.to(device)
    model.eval()

    # Perform specific preprocessing for the model
    preprocess = weights.transforms()

    # Get the patches
    # patch_scale = 1.0
    flat_patches = adata.obsm[f'patches_scale_{patch_scale}']

    # Reshape all the patches to the original shape
    all_patches = flat_patches.reshape((-1, patch_size, patch_size, 3))
    torch_patches = torch.from_numpy(all_patches).permute(0, 3, 1, 2).float()    # Turn all patches to torch
    rescaled_patches = torch_patches / 255                                       # Rescale patches to [0, 1]
    processed_patches = preprocess(rescaled_patches)                             # Preprocess patches
    
    # Create a dataloader
    dataloader = DataLoader(processed_patches, batch_size=256, shuffle=False, num_workers=4)

    # Declare lists to store the embeddings or predictions
    outputs = []

    with torch.no_grad():
        desc = 'Getting predictions'
        
        for batch in tqdm(dataloader, desc=desc):
            batch = batch.to(device)                    # Send batch to device                
            batch_output = model(batch)                 # Get predictions
            outputs.append(batch_output)                # Append to list


    # Concatenate all embeddings or predictions
    outputs = torch.cat(outputs, dim=0)

    # Pass predictions to cpu and add to data.obsm
    adata.obsm[f'predictions_{backbone}'] = outputs.cpu().numpy()

### Define function to get dimensionality reductions depending on the layer
def compute_dim_red(adata: ad.AnnData, from_layer: str) -> ad.AnnData:
    """ Compute embeddings and clusters

    Simple wrapper around sc.pp.pca, sc.pp.neighbors, sc.tl.umap and sc.tl.leiden to compute the embeddings and cluster the data.
    Everything will be computed using the expression matrix stored in adata.layers[from_layer]. 

    Args:
        adata (ad.AnnData): The AnnData object to transform. Must have expression values in adata.layers[from_layer].
        from_layer (str): The key in adata.layers where the expression matrix is stored.

    Returns:
        ad.AnnData: The transformed AnnData object with the embeddings and clusters.
    """
    
    # Start the timer
    # start = time()
    # print(f'Computing embeddings and clusters using data of layer {from_layer}...')
    
    # Set the key layer as the main expression matrix
    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers[from_layer]
    

    # Compute the embeddings and clusters
    sc.pp.pca(adata_copy, random_state=42)
    sc.pp.neighbors(adata_copy, random_state=42)
    sc.tl.umap(adata_copy, random_state=42)
    sc.tl.leiden(adata_copy, key_added="cluster", random_state=42)
    
    # Restore the original expression matrix as counts layer
    adata_copy.X = adata_copy.layers['counts']

    # Print the time it took to compute the embeddings and clusters
    # print(f'Embeddings and clusters computed in {time() - start:.2f} seconds')

    # Return the adapted AnnData object
    return adata_copy

### Define function to get spatial neighbors in an AnnData object
def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
    """ Compute neighbors dictionary for an AnnData object.
    
    This function computes a neighbors dictionary for an AnnData object. The neighbors are computed according to topological distances over
    a graph defined by the hex_geometry connectivity. The neighbors dictionary is a dictionary where the keys are the indexes of the observations
    and the values are lists of the indexes of the neighbors of each observation. The neighbors include the observation itself and are found
    inside an n_hops neighborhood (vicinity) of the observation.

    Args:
        adata (ad.AnnData): The AnnData object to process. Importantly it is only from a single slide. Can not be a collection of slides.
        n_hops (int): The size of the neighborhood to take into account to compute the neighbors.
        hex_geometry (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only
                                true for visium datasets.

    Returns:
        dict: The neighbors dictionary. The keys are the indexes of the observations and the values are lists of the indexes of the neighbors of each observation.
    """

    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
    else:
        sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid dataset case

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