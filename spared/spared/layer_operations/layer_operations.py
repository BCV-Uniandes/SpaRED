import anndata as ad
import os
import torch
from time import time
import numpy as np
import pandas as pd
import pathlib
import shutil
import wget
import gzip
import subprocess
from combat.pycombat import pycombat
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import sys

# Path a spared 
SPARED_PATH = pathlib.Path(__file__).resolve().parent
# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from spared.denoising import denoising
from spared.gene_features import gene_features
from spared.filtering import filtering
from spared.layer_operations import layer_operations
#Remove the path from sys.path

### Expression data processing functions:
def tpm_normalization(adata: ad.AnnData, organism: str, from_layer: str, to_layer: str) -> ad.AnnData:
    """Normalize expression using TPM normalization.

    This function applies TPM normalization to an AnnData object with raw counts. It also removes genes that are not fount in the ``.gtf`` annotation file.
    The counts are taken from ``adata.layers[from_layer]`` and the results are stored in ``adata.layers[to_layer]``. It can perform the normalization
    for `human <https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz>`_ and `mouse
    <https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.basic.annotation.gtf.gz>`_ reference genomes.
    To specify which GTF annotation file should be used, the ``'organism'`` parameter must be ``'mouse'`` or ``'human'``.

    Args:
        adata (ad.Anndata): The Anndata object to normalize.
        organism (str): Organism of the dataset. Must be 'mouse' or 'human'.
        from_layer (str): The layer to take the counts from. The data in this layer should be in raw counts.
        to_layer (str): The layer to store the results of the normalization.
    Returns:
        ad.Anndata: The updated Anndata object with TPM values in ``adata.layers[to_layer]``.
    """
    
    # Get the number of genes before filtering
    SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
    print(SPARED_PATH)
    initial_genes = adata.shape[1]

    # Automatically download the human gtf annotation file if it is not already downloaded
    if not os.path.exists(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz')):
        print('Automatically downloading human gtf annotation file...')
        os.makedirs(os.path.join(SPARED_PATH, 'data', 'annotations'), exist_ok=True)
        wget.download(
            'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.basic.annotation.gtf.gz',
            out = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'))

    # Define gtf human path
    gtf_path = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf')

    # Unzip the data in annotations folder if it is not already unzipped
    if not os.path.exists(gtf_path):
        with gzip.open(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.v43.basic.annotation.gtf.gz'), 'rb') as f_in:
            with open(gtf_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # FIXME: Set up automatic download of mouse gtf file (DONE)
    # Automatically download the mouse gtf annotation file if it is not already downloaded
    if not os.path.exists(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf.gz')):
        print('Automatically downloading mouse gtf annotation file...')
        os.makedirs(os.path.join(SPARED_PATH, 'data', 'annotations'), exist_ok=True)
        wget.download(
            'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.basic.annotation.gtf.gz',
            out = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf.gz'))
    
    # Define gtf mouse path
    gtf_path_mouse = os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf')

    # Unzip the data in annotations folder if it is not already unzipped
    if not os.path.exists(gtf_path_mouse):            
        with gzip.open(os.path.join(SPARED_PATH, 'data', 'annotations', 'gencode.vM33.basic.annotation.gtf.gz'), 'rb') as f_in:
            with open(gtf_path_mouse, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    # Obtain a txt with gene lengths
    gene_length_path = os.path.join(SPARED_PATH, 'data', 'annotations', 'gene_length.txt')
    if not os.path.exists(gene_length_path):
        command = f'python {os.path.join(SPARED_PATH, "gtftools.py")} -l {gene_length_path} {gtf_path}'
        command_list = command.split(' ')
        subprocess.call(command_list)   

    gene_length_path_mouse = os.path.join(SPARED_PATH, 'data', 'annotations', 'gene_length_mouse.txt')
    if not os.path.exists(gene_length_path_mouse):
        command = f'python {os.path.join(SPARED_PATH, "gtftools.py")} -l {gene_length_path_mouse} {gtf_path_mouse}'
        command_list = command.split(' ')
        subprocess.call(command_list) 

    # Upload the gene lengths
    if organism.lower() == "mouse":
        glength_df = pd.read_csv(gene_length_path_mouse, delimiter='\t', usecols=['gene', 'merged'])
    elif organism.lower() == "human":
        glength_df = pd.read_csv(gene_length_path, delimiter='\t', usecols=['gene', 'merged'])
    else:
        assert "Organism not valid"

    # For the gene column, remove the version number
    glength_df['gene'] = glength_df['gene'].str.split('.').str[0]

    # Drop gene duplicates. NOTE: This only eliminates 40/60k genes so it is not a big deal
    glength_df = glength_df.drop_duplicates(subset=['gene'])

    # Find the genes that are in the gtf annotation file
    common_genes=list(set(adata.var_names)&set(glength_df["gene"]))

    # Subset both adata and glength_df to keep only the common genes
    adata = adata[:, common_genes].copy()
    glength_df = glength_df[glength_df["gene"].isin(common_genes)].copy()

    # Reindex the glength_df to genes
    glength_df = glength_df.set_index('gene')
    # Reindex glength_df to adata.var_names
    glength_df = glength_df.reindex(adata.var_names)
    # Assert indexes of adata.var and glength_df are the same
    assert (adata.var.index == glength_df.index).all()

    # Add gene lengths to adata.var
    adata.var['gene_length'] = glength_df['merged'].values

    # Divide each column of the counts matrix by the gene length. Save the result in layer "to_layer"
    adata.layers[to_layer] = adata.layers[from_layer] / adata.var['gene_length'].values.reshape(1, -1)
    # Make that each row sums to 1e6
    adata.layers[to_layer] = np.nan_to_num(adata.layers[to_layer] / (np.sum(adata.layers[to_layer], axis=1).reshape(-1, 1)/1e6))
    # Pass layer to np.array
    adata.layers[to_layer] = np.array(adata.layers[to_layer])

    # Print the number of genes that were not found in the gtf annotation file
    failed_genes = initial_genes - adata.n_vars
    print(f'Number of genes not found in GTF file by TPM normalization: {initial_genes - adata.n_vars} out of {initial_genes} ({100*failed_genes/initial_genes:.2f}%) ({adata.n_vars} remaining)')

    # Return the transformed AnnData object
    return adata

def log1p_transformation(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
    """Perform :math:`\log_2(x+1)` transformation 

    Performs logarithmic transformation over ``adata.layers[from_layer]``. Simple wrapper of scanpy's ``sc.pp.log1p()``
    (base 2) to transform data from ``adata.layers[from_layer]`` and save it into ``adata.layers[to_layer]``.

    Args:
        adata (ad.AnnData): The AnnData object to transform.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with transformed data in ``adata.layers[to_layer]``.
    """

    # Transform the data with log1p
    transformed_adata = sc.pp.log1p(adata, base= 2.0, layer=from_layer, copy=True)

    # Add the log1p transformed data to adata.layers[to_layer]
    adata.layers[to_layer] = transformed_adata.layers[from_layer]

    # Return the transformed AnnData object
    return adata

# FIXME: Update to the new hosting of the pycombat package
def combat_transformation(adata: ad.AnnData, batch_key: str, from_layer: str, to_layer: str) -> ad.AnnData:
    """ Perform batch correction with ComBat

    Compute batch correction using the `pycombat <https://github.com/epigenelabs/pyComBat?tab=readme-ov-file>`_ package. The batches are defined by ``adata.obs[batch_key]`` so
    the user can define which variable to use as batch identifier. The input data for the batch correction is ``adata.layers[from_layer]`` and the output is stored in
    ``adata.layers[to_layer]``. Importantly, as the `original ComBat paper <https://doi.org/10.1093/biostatistics/kxj037>`_ notes the data should be approximately normally distributed. Therefore, it is recommended to use
    this function over :math:`\log_2(TPM+1)` data.

    Args:
        adata (ad.AnnData): The AnnData object to transform. Must have logarithmically transformed data in ``adata.layers[from_layer]``.
        batch_key (str): The column in ``adata.obs`` that defines the batches.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with batch corrected data in ``adata.layers[to_layer]``.
    """
    # Get expression matrix dataframe
    df = adata.to_df(layer = from_layer).T
    batch_list = adata.obs[batch_key].values.tolist()

    # Apply pycombat batch correction
    corrected_df = pycombat(df, batch_list, par_prior=True)

    # Assign batch corrected expression to .layers[to_layer] attribute
    adata.layers[to_layer] = corrected_df.T

    return adata

# TODO: Put reference to SEPAL as the first method trying this
def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
    """ Get expression deltas from the mean.

    Compute the deviations from the mean expression of each gene in ``adata.layers[from_layer]`` and save them
    in ``adata.layers[to_layer]``. Also add the mean expression of each gene to ``adata.var[f'{from_layer}_avg_exp']``.
    Average expression is computed using only train data determined by the ``adata.obs['split']`` column. However, deltas
    are computed for all observations.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have expression values in ``adata.layers[from_layer]``. Must also have the ``adata.obs['split']`` column with ``'train'`` values.
        from_layer (str): The layer to take the data from.
        to_layer (str): The layer to store the results of the transformation.

    Returns:
        ad.AnnData: The updated AnnData object with the deltas (``adata.layers[to_layer]``) and mean expression (``adata.var[f'{from_layer}_avg_exp']``) information.
    """

    # Get the expression matrix of both train and global data
    glob_expression = adata.to_df(layer=from_layer)
    train_expression = adata[adata.obs['split'] == 'train'].to_df(layer=from_layer)

    # Define scaler
    scaler = StandardScaler(with_mean=True, with_std=False)

    # Fit the scaler to the train data
    scaler = scaler.fit(train_expression)
    
    # Get the centered expression matrix of the global data
    centered_expression = scaler.transform(glob_expression)

    # Add the deltas to adata.layers[to_layer]	
    adata.layers[to_layer] = centered_expression

    # Add the mean expression to adata.var[f'{from_layer}_avg_exp']
    adata.var[f'{from_layer}_avg_exp'] = scaler.mean_

    # Return the updated AnnData object
    return adata

# TODO: modify function to add noisy layer to important layers
def add_noisy_layer(adata: ad.AnnData, prediction_layer: str) -> ad.AnnData:
    """ Add an artificial noisy layer.

    This function should only be used for experimentation/ablation purposes. The noisy layer is created by returning the missing values to an already denoised
    layer of the ``adata``. In the case the ``'prediction_layer'`` is on :math:`\log_2(TPM+1)` logarithmic scale, the noisy layer is created by assigning zero values
    to the missing values (adds ``'noisy'`` layer to the adata). In the case the ``'prediction_layer'`` is on delta scale, the noisy layer is created by assigning the
    negative mean of the gene to the missing values (adds ``'noisy_d'`` layer to the adata). Missing values are specified by the binary ``adata.layers['mask']``
    layer that must be already present and has ``True`` values for all real data and ``False`` values for imputed data.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have the ``adata.layers[prediction_layer]``, the gene means if its a delta layer, and ``adata.layers['mask']``.
        prediction_layer (str): The layer that will be corrupted to create the noisy layer.

    Returns:
        ad.AnnData: The updated AnnData object with the ``adata.layers['noisy']`` or ``adata.layers['noisy_d']`` layer added depending on ``prediction_layer``.
    """
    
    if 'delta' in prediction_layer:
        # Get vector with gene means
        avg_layer = prediction_layer.replace("deltas","log1p")
        gene_means = adata.var[f"{avg_layer}_avg_exp"].values 
        # Expand gene means to the shape of the layer
        gene_means = np.repeat(gene_means.reshape(1, -1), adata.n_obs, axis=0)
        # Get valid mask
        valid_mask = adata.layers['mask']
        # Initialize noisy deltas
        noisy_deltas = -gene_means 
        delta_key = prediction_layer.split("log1p")
        # Assign delta values in positions where valid mask is true
        noisy_deltas[valid_mask] = adata.layers[prediction_layer][valid_mask]
        # Add the layer to the adata
        adata.layers[f'noisy_{prediction_layer}'] = noisy_deltas

        # Add a var column of used means of the layer
        mean_key = f'{avg_layer}_avg_exp'
        adata.var[f'used_mean_{prediction_layer}'] = adata.var[mean_key]

    else:
        # Copy the cleaned layer to the layer noisy
        noised_layer = adata.layers[prediction_layer].copy()
        # Get zero mask
        zero_mask = ~adata.layers['mask']
        # Zero out the missing values
        noised_layer[zero_mask] = 0
        # Add the layer to the adata
        adata.layers[f'noisy_{prediction_layer}'] = noised_layer

    return adata

# TODO: Put reference to the filter_dataset() function
# FIXME: Here we should have the transformer cleaner too
# TODO: Add combat link
def process_dataset(adata: ad.AnnData, param_dict: dict, dataset: str) -> ad.AnnData:
    """ Perform complete processing pipeline.

    This function performs the complete processing pipeline. It only computes over the expression and filters genes to get the final prediction
    variables. However, it doesn't perform spot (sample) filtering for which the ``filter_dataset()`` function is recommended. The input data
    ``adata.X`` is expected to be in raw counts. The processing pipeline is the following:

        1. Normalize the data with TPM normalization (adds ``adata.layers['tpm']``)
        2. Transform the data with logarithmically using :math:`\log_2(TPM+1)` (adds ``adata.layers['log1p']``)
        3. Denoise the data with the adaptive median filter (adds ``adata.layers['d_log1p']``)
        4. Compute Moran's I for each gene in each slide and average Moran's I across slides (adds ``adata.var['d_log1p_moran']``)
        5. Filter dataset to keep the top ``param_dict['top_moran_genes']`` genes with highest Moran's I.
        6. Perform ComBat batch correction if specified by the ``param_dict['combat_key']`` parameter (adds ``adata.layers['c_d_log1p']``)
        7. Compute the deltas from the mean for each gene. Computed from ``log1p``, ``d_log1p`` and ``c_log1p``, ``c_d_log1p`` layer if batch correction was performed (adds ``deltas``, ``d_deltas``, ``c_deltas``, ``c_d_deltas`` layers) 
        8. Add a binary mask layer specifying valid observations for metric computation (adds ``adata.layers['mask']``, ``True`` for valid observations, ``False`` for missing values).

    Args:
        adata (ad.AnnData): The AnnData object to process. Should be already spot/sample filtered..
        param_dict (dict): Dictionary that contains filtering and processing parameters. Keys that must be present are:
                            - 'top_moran_genes': (int) The number of genes to keep after filtering by Moran's I. If set to 0, then the number of genes is internally computed.
                            - 'combat_key': (str) The column in adata.obs that defines the batches for ComBat batch correction. If set to 'None', then no batch correction is performed.
                            - "hex_geometry" (bool): Whether the graph is hexagonal or not. If True, then the graph is hexagonal. If False, then the graph is a grid. Only true for visium datasets.
        dataset_name (str): The name of the dataset.
        
    Returns:
        ad.Anndata: The processed AnnData object with all the layers and results added. A list of included keys in ``adata.layers`` is:

                    - ``'counts'``: Raw counts of the dataset.
                    - ``'tpm'``: TPM normalized data.
                    - ``'log1p'``: :math:`\log_2(TPM+1)` transformed data.
                    - ``'d_log1p'``: Denoised data with adaptive median filter.
                    - ``'c_log1p'``: Batch corrected data with ComBat (only if ``param_dict['combat_key'] != 'None'``).
                    - ``'c_d_log1p'``: Batch corrected and denoised data with adaptive median filter (only if ``param_dict['combat_key'] != 'None'``).
                    - ``'deltas'``: Deltas from the mean expression for ``log1p``.
                    - ``'d_deltas'``: Deltas from the mean expression for ``d_log1p``.
                    - ``'c_deltas'``: Deltas from the mean expression for ``c_log1p`` (only if ``param_dict['combat_key'] != 'None'``).
                    - ``'c_d_deltas'``: Deltas from the mean expression for ``c_d_log1p`` (only if ``param_dict['combat_key'] != 'None'``).
                    - ``'noisy_c_d_log1p'``: Processed layer c_d_log1p where original missing values are replaced with 0.
                    - ``'noisy_c_d_deltas'``: Processed layer c_d_deltas where original missing values are replaced with the negative mean expression of the gene.
                    - ``'noisy_c_t_log1p'``: Processed layer c_t_log1p where original missing values are replaced with 0.
                    - ``'noisy_c_t_deltas'``: Processed layer c_t_deltas where original missing values are replaced with the negative mean expression of the gene.
                    - ``'mask'``: Binary mask layer. ``True`` for valid observations, ``False`` for imputed missing values.
    """

    ### Compute all the processing steps
    # NOTE: The d prefix stands for denoised
    # NOTE: The c prefix stands for combat corrected

    # Start the timer and print the start message
    start = time()
    print('Starting data processing...')

    # First add raw counts to adata.layers['counts']
    adata.layers['counts'] = adata.X.toarray()
    
    # 1. Make TPM normalization
    adata = tpm_normalization(adata, param_dict["organism"], from_layer='counts', to_layer='tpm')

    # 2. Transform the data with log1p (base 2.0)
    adata = log1p_transformation(adata, from_layer='tpm', to_layer='log1p')

    # 3. Denoise the data with adaptive median filter
    adata = denoising.median_cleaner(adata, from_layer='log1p', to_layer='d_log1p', n_hops=4, hex_geometry=param_dict["hex_geometry"])

    # 4. Compute average moran for each gene in the layer d_log1p 
    adata = gene_features.compute_moran(adata, hex_geometry=param_dict["hex_geometry"], from_layer='d_log1p')

    # 5. Filter genes by Moran's I
    adata = filtering.filter_by_moran(adata, n_keep=param_dict['top_moran_genes'], from_layer='d_log1p')

    # 6. If combat key is specified, apply batch correction
    if param_dict['combat_key'] != 'None':
        adata = combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
        adata = combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='d_log1p', to_layer='c_d_log1p')

    # 7. Compute deltas and mean expression for all log1p, d_log1p, c_log1p and c_d_log1p
    adata = get_deltas(adata, from_layer='log1p', to_layer='deltas')
    adata = get_deltas(adata, from_layer='d_log1p', to_layer='d_deltas')
    
    if param_dict['combat_key'] != 'None':
        adata = get_deltas(adata, from_layer='c_log1p', to_layer='c_deltas')
        adata = get_deltas(adata, from_layer='c_d_log1p', to_layer='c_d_deltas')

    # 8. Add a binary mask layer specifying valid observations for metric computation
    adata.layers['mask'] = adata.layers['tpm'] != 0
    
    # Define a device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 9. Denoise the data with spackle method
    adata, _  = denoising.spackle_cleaner(adata=adata, dataset = dataset, from_layer="c_d_log1p", to_layer="c_t_log1p", device = device)
    
    # 10. Get delta for c_t_log1p
    adata = get_deltas(adata, from_layer='c_t_log1p', to_layer='c_t_deltas')
    
    # 11. Add noisy layers 
    list_layers = ["c_d_log1p",
                   "c_t_log1p",
                   "c_d_deltas",
                   "c_t_deltas"]
    
    for layer in list_layers:
        adata = layer_operations.add_noisy_layer(adata=adata, prediction_layer=layer)
    
    # Print with the percentage of the dataset that was replaced
    print('Percentage of imputed observations with median filter and spackle method: {:5.3f}%'.format(100 * (~adata.layers["mask"]).sum() / (adata.n_vars*adata.n_obs)))

    # Print the number of cells and genes in the dataset
    print(f'Processing of the data took {time() - start:.2f} seconds')
    print(f'The processed dataset looks like this:')
    print(adata)
    
    return adata

