import anndata as ad
from time import time
import numpy as np
import pandas as pd
import pathlib
import scanpy as sc
import sys

# El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from gene_features import gene_features
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

def filter_by_moran(adata: ad.AnnData, n_keep: int, from_layer: str) -> ad.AnnData:
    """ Filter prediction genes by Moran's I.

    This function filters the genes in ``adata.var`` by the Moran's I statistic. It keeps the ``n_keep`` genes with the highest Moran's I.
    The Moran's I values will be selected from ``adata.var[f'{from_layer}_moran']`` which must be already present in the ``adata``.
    If ``n_keep <= 0``, it means the number of genes is no specified and wee proceed to automatically compute it in the following way:
    
        a. If ``adata.n_vars > 320`` then ``n_keep = 128``.
        b. else, ``n_keep = 32``. 

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have ``adata.var[f'{from_layer}_moran']`` column.
        n_keep (int): The number of genes to keep. I less than ``0`` the number of genes to keep is computed automatically.
        from_layer (str): Layer for which the Moran's I was already computed (``adata.var[f'{from_layer}_moran']``).

    Returns:
        ad.AnnData: The updated AnnData object with the filtered genes.
    """

    # Assert that the number of genes is at least n_keep
    assert adata.n_vars >= n_keep, f'The number of genes in the AnnData object is {adata.n_vars}, which is less than n_keep ({n_keep}).'

    # FIXME: This part is weird, we can define a simple threshold without all the computation (DONE)
    # threshold: 320
    if n_keep <= 0:
        if adata.n_vars > 320:
            n_keep = 128
        else:
            n_keep = 32

    print(f"Filtering genes by Moran's I. Keeping top {n_keep} genes.")
    
    # Sort the genes by Moran's I
    sorted_genes = adata.var.sort_values(by=f'{from_layer}_moran', ascending=False).index

    # Get genes to keep list
    genes_to_keep = list(sorted_genes[:n_keep])

    # Filter the genes andata object
    adata = adata[:, genes_to_keep]

    # Return the updated AnnData object
    return adata

def filter_dataset(adata: ad.AnnData, param_dict: dict) -> ad.AnnData:
    """ Perform complete filtering pipeline of a slide collection.

    This function takes a completely unfiltered and unprocessed (in raw counts) slide collection and filters it
    (both samples and genes) according to the ``param_dict`` argument.
    A summary list of the steps is the following:

        1. Filter out observations with ``total_counts`` outside the range ``[param_dict['cell_min_counts'], param_dict['cell_max_counts']]``.
           This filters out low quality observations not suitable for analysis.
        2. Compute the ``exp_frac`` for each gene. This means that for each slide in the collection we compute the fraction of the spots that express each gene and then take the minimum across all the slides (see ``get_exp_frac`` function for more details).
        3. Compute the ``glob_exp_frac`` for each gene. This is similar to the ``exp_frac`` but instead of computing for each
           slide and taking the minimum we compute it for the whole collection. Slides don't matter here
           (see ``get_glob_exp_frac`` function for more details).
        4. Filter out genes depending on the ``param_dict['wildcard_genes']`` value, the options are the following:

            a. ``param_dict['wildcard_genes'] == 'None'``:

                - Filter out genes that are not expressed in at least ``param_dict['min_exp_frac']`` of spots in each slide.
                - Filter out genes that are not expressed in at least ``param_dict['min_glob_exp_frac']`` of spots in the whole collection.
                - Filter out genes with counts outside the range ``[param_dict['gene_min_counts'], param_dict['gene_max_counts']]``
            b. ``param_dict['wildcard_genes'] != 'None'``:

                - Read ``.txt`` file specified by ``param_dict['wildcard_genes']`` and leave only the genes that are in this file.
        5. If there are spots with zero counts in all genes after gene filtering, remove them.
        6. Compute quality control metrics using scanpy's ``sc.pp.calculate_qc_metrics`` function.

    Args:
        adata (ad.AnnData): An unfiltered (unexpressed genes are encoded as ``0`` on the ``adata.X matrix``) slide collection.
        param_dict (dict): Dictionary that contains filtering and processing parameters. Keys that must be present are:

            - ``'cell_min_counts'`` (*int*):      Minimum total counts for a spot to be valid.
            - ``'cell_max_counts'`` (*int*):      Maximum total counts for a spot to be valid.
            - ``'gene_min_counts'`` (*int*):      Minimum total counts for a gene to be valid.
            - ``'gene_max_counts'`` (*int*):      Maximum total counts for a gene to be valid.
            - ``'min_exp_frac'`` (*float*):       Minimum fraction of spots in any slide that must express a gene for it to be valid.
            - ``'min_glob_exp_frac'`` (*float*):  Minimum fraction of spots in the whole collection that must express a gene for it to be valid.
            - ``'wildcard_genes'`` (*str*):       Path to a ``.txt`` file with the genes to keep or ``'None'`` to filter genes based on the other keys.

    Returns:
        ad.AnnData: The filtered adata collection.
    """

    # Start tracking time
    print('Starting data filtering...')
    start = time()

    # Get initial gene and observation numbers
    n_genes_init = adata.n_vars
    n_obs_init = adata.n_obs

    ### Filter out samples:

    # Find indexes of cells with total_counts outside the range [cell_min_counts, cell_max_counts]
    sample_counts = np.squeeze(np.asarray(adata.X.sum(axis=1)))
    bool_valid_samples = (sample_counts > param_dict['cell_min_counts']) & (sample_counts < param_dict['cell_max_counts'])
    valid_samples = adata.obs_names[bool_valid_samples]

    # Subset the adata to keep only the valid samples
    adata = adata[valid_samples, :].copy()

    ### Filter out genes:

    # Compute the min expression fraction for each gene across all the slides
    adata = gene_features.get_exp_frac(adata)
    # Compute the global expression fraction for each gene
    adata = gene_features.get_glob_exp_frac(adata)
    
    # If no wildcard genes are specified then filter genes based in min_exp_frac and total counts
    if param_dict['wildcard_genes'] == 'None':
        
        gene_counts = np.squeeze(np.asarray(adata.X.sum(axis=0)))
                    
        # Find indexes of genes with total_counts inside the range [gene_min_counts, gene_max_counts]
        bool_valid_gene_counts = (gene_counts > param_dict['gene_min_counts']) & (gene_counts < param_dict['gene_max_counts'])
        # Get the valid genes
        valid_genes = adata.var_names[bool_valid_gene_counts]
        
        # Subset the adata to keep only the valid genes
        adata = adata[:, valid_genes].copy()     
    
        # Filter by expression fractions - order by descending expression fraction
        df_exp = adata.var.copy().sort_values('exp_frac', ascending=False)
        # Calculate the mean glob_exp_frac of top expression fraction genes
        df_exp['Row'] = range(1, len(df_exp) + 1)
        df_exp['vol_real_data'] = df_exp['glob_exp_frac'].cumsum() / (df_exp['Row'])      
        df_exp = df_exp.drop(['Row'], axis=1)
        # Get the valid genes
        num_genes = np.where(df_exp['vol_real_data'] >= param_dict['real_data_percentage'])[0][-1]
        valid_genes = df_exp.iloc[:num_genes + 1]['gene_ids']
        # Subset the adata to keep only the valid genes
        adata = adata[:, valid_genes].copy()
    
    # If there are wildcard genes then read them and subset the dataset to just use them
    else:
        # Read valid wildcard genes
        genes = pd.read_csv(param_dict['wildcard_genes'], sep=" ", header=None, index_col=False)
        # Turn wildcard genes to pandas Index object
        valid_genes = pd.Index(genes.iloc[:, 0], name='')
        # Subset processed adata with wildcard genes
        adata = adata[:, valid_genes].copy()
    
    ### Remove cells with zero counts in all genes:

    # If there are cells with zero counts in all genes then remove them
    null_cells = adata.X.sum(axis=1) == 0
    if null_cells.sum() > 0:
        adata = adata[~null_cells].copy()
        print(f"Removed {null_cells.sum()} cells with zero counts in all selected genes")
    
    ### Compute quality control metrics:

    # As we have removed the majority of the genes, we recompute the quality metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False, percent_top=None)

    # Print the number of genes and cells that survived the filtering
    print(f'Data filtering took {time() - start:.2f} seconds')
    print(f"Number of genes that passed the filtering:        {adata.n_vars} out of {n_genes_init} ({100*adata.n_vars/n_genes_init:.2f}%)")
    print(f"Number of observations that passed the filtering: {adata.n_obs} out of {n_obs_init} ({100*adata.n_obs/n_obs_init:.2f}%)")

    return adata

def get_slide_from_collection(collection: ad.AnnData,  slide: str) -> ad.AnnData:
    """ Retrieve a slide from a collection of slides.

    This function receives a slide name and returns an AnnData object of the specified slide based on the collection of slides
    in the collection parameter.

    Args: 
        collection (ad.AnnData): AnnData object with all the slides concatenated.
        slide (str): Name of the slide to get from the collection. Must be in the ``slide_id`` column of the ``collection.obs`` dataframe.

    Returns:
        ad.AnnData: An AnnData object with the specified slide.
    """

    # Get the slide from the collection
    slide_adata = collection[collection.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: collection.uns['spatial'][slide]}

    # Return the slide
    return slide_adata

def get_slides_adata(collection: ad.AnnData, slide_list: str) -> list:
    """ Get list of adatas to plot

    This function receives a string with a list of slides separated by commas and returns a list of anndata objects with
    the specified slides taken from the collection parameter. 

    Args:
        collection (ad.AnnData): Processed and filtered data ready to use by the model.
        slide_list (str): String with a list of slides separated by commas.

    Returns:
        list: List of anndata objects with the specified slides.
    """

    # Get the slides from the collection
    #s_adata_list = [self.get_slide_from_collection(collection,  slide) for slide in slide_list.split(',')]
    s_adata_list = []
    
    for slide in slide_list.split(','):  
        # Get the slide from the collection
        slide_adata = collection[collection.obs['slide_id'] == slide].copy()
        # Modify the uns dictionary to include only the information of the slide
        slide_adata.uns['spatial'] = {slide: collection.uns['spatial'][slide]}
        s_adata_list.append(slide_adata)

    # Return the slides
    return s_adata_list


