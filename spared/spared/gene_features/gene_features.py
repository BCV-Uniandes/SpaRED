import anndata as ad
import numpy as np
import pandas as pd
import pathlib
import squidpy as sq
import sys

# El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from filtering import filtering
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
    """ Compute the expression fraction for all genes.

    The expression fraction of a gene in a slide is defined as the proportion of spots where that gene is expressed. It is a number between ``0``
    and ``1`` where ``0`` means that the gene is not expressed in any spot and ``1`` means that the gene is expressed in all the spots.

    To compute an aggregation of expression fractions in a complete dataset, this function gets the
    expression fraction for each slide and then takes the minimum across all the slides. Hence the final number is a lower bound that ensures
    that the gene is expressed in at least that fraction of the spots in each of the slides.

    Args:
        adata (ad.AnnData): A slide collection where non-expressed genes have a value of ``0`` in the ``adata.X`` matrix.

    Returns:
        ad.AnnData: The updated slide collection with the added information into the ``adata.var['exp_frac']`` column.
    """
    # Get the unique slide ids
    slide_ids = adata.obs['slide_id'].unique()

    # Define zeros matrix of shape (n_genes, n_slides)
    exp_frac = np.zeros((adata.n_vars, len(slide_ids)))

    # Iterate over the slide ids
    for i, slide_id in enumerate(slide_ids):
        # Get current slide adata
        slide_adata = adata[adata.obs['slide_id'] == slide_id, :]
        # Get current slide expression fraction
        curr_exp_frac = np.squeeze(np.asarray((slide_adata.X > 0).sum(axis=0) / slide_adata.n_obs))
        # Add current slide expression fraction to the matrix
        exp_frac[:, i] = curr_exp_frac
    
    # Compute the minimum expression fraction for each gene across all the slides
    min_exp_frac = np.min(exp_frac, axis=1)

    # Add the minimum expression fraction to the var dataframe of the slide collection
    adata.var['exp_frac'] = min_exp_frac

    # Return the adata
    return adata

def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
    """ Compute the global expression fraction for all genes.
    
    This function computes the global expression fraction for each gene in a dataset.

    The global expression fraction of a gene in a dataset is defined as the proportion of spots where that gene is expressed. It is a number between ``0``
    and ``1`` where ``0`` means that the gene is not expressed in any spot and ``1`` means that the gene is expressed in all the spots. Its difference
    with the expression fraction is that the global expression fraction is computed for the whole dataset and not for each slide.

    Args:
        adata (ad.AnnData): A slide collection where a non-expressed genes have a value of ``0`` in the ``adata.X`` matrix.

    Returns:
        ad.AnnData: The updated slide collection with the information added into the  ``adata.var['glob_exp_frac']`` column.
    """
    # Get global expression fraction
    glob_exp_frac = np.squeeze(np.asarray((adata.X > 0).sum(axis=0) / adata.n_obs))

    # Add the global expression fraction to the var dataframe of the slide collection
    adata.var['glob_exp_frac'] = glob_exp_frac

    # Return the adata
    return adata

def compute_moran(adata: ad.AnnData, from_layer: str, hex_geometry: bool) -> ad.AnnData:
    
    """Compute Moran's I statistic for each gene.

    Compute average Moran's I statistic for a collection of slides. Internally cycles over each slide in the ``adata`` collection
    and computes the Moran's I statistic for each gene. After that, it averages the Moran's I for each gene across all
    slides and saves it in ``adata.var[f'{from_layer}_moran']``.The input data for the Moran's I computation is ``adata.layers[from_layer]``.

    Args:
        adata (ad.AnnData): The AnnData object to update. Must have expression values in ``adata.layers[from_layer]``.
        from_layer (str): The key in ``adata.layers`` with the values used to compute Moran's I.
        hex_geometry (bool): Whether the geometry is hexagonal or not. This is used to compute the spatial neighbors before computing Moran's I. Only ``True`` for visium datasets.

    Returns:
        ad.AnnData: The updated AnnData object with the average Moran's I for each gene in ``adata.var[f'{from_layer}_moran']``.
    """
    print(f'Computing Moran\'s I for each gene over each slide using data of layer {from_layer}...')

    # Get the unique slide_ids
    slide_ids = adata.obs['slide_id'].unique()

    # Create a dataframe to store the Moran's I for each slide
    moran_df = pd.DataFrame(index = adata.var.index, columns=slide_ids)

    # Cycle over each slide
    for slide in slide_ids:
        # Get the annData for the current slide
        slide_adata = filtering.get_slide_from_collection(adata, slide)
        # Compute spatial_neighbors
        if hex_geometry:
            # Hexagonal visium case
            sq.gr.spatial_neighbors(slide_adata, coord_type='generic', n_neighs=6)
        else:
            # Grid STNet dataset case
            sq.gr.spatial_neighbors(slide_adata, coord_type='grid', n_neighs=8)
        # Compute Moran's I
        sq.gr.spatial_autocorr(
            slide_adata,
            mode="moran",
            layer=from_layer,
            genes=slide_adata.var_names,
            n_perms=1000,
            n_jobs=-1,
            seed=42
        )

        # Get moran I
        moranI = slide_adata.uns['moranI']['I']
        # Reindex moranI to match the order of the genes in the adata object
        moranI = moranI.reindex(adata.var.index)

        # Add the Moran's I to the dataframe
        moran_df[slide] = moranI

    # Compute the average Moran's I for each gene
    adata.var[f'{from_layer}_moran'] = moran_df.mean(axis=1)

    # Return the updated AnnData object
    return adata


