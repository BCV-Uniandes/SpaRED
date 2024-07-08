import scanpy as sc
import anndata as ad
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap   
import matplotlib.colors as colors
from time import time
import pathlib
import sys

#Path a spared
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

#Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
#import requiere files
from gene_features import gene_features
from filtering import filtering
from spot_features import spot_features

def plot_all_slides(dataset: str, processed_adata: ad.AnnData, path: str) -> None:
    """ Plot all the whole slide images

    This function takes a slide collection and plot all the whole slide images in a square aspect ratio.

    Args:
        dataset: Name of the dataset
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        path (str): Path to save the plot.
    """
    
    # Get unique slide ids
    unique_ids = sorted(processed_adata.obs['slide_id'].unique())

    # Get number of rows and columns for the number of slides
    n = int(np.ceil(np.sqrt(len(unique_ids))))
    m = int(np.ceil(float(len(unique_ids))/float(n)))

    # Define figure
    fig, ax = plt.subplots(nrows=m, ncols=n)
    fig.set_size_inches(7, (7*m/n)+1)
    
    # Flatten axes list
    ax = ax.flatten()

    for i, curr_id in enumerate(unique_ids):
        curr_img = processed_adata.uns['spatial'][curr_id]['images']['lowres']
        ax[i].imshow(curr_img)
        ax[i].set_title(curr_id, fontsize='large')

    # Remove axis for all the figure
    [axis.axis('off') for axis in ax]
    fig.suptitle(f'All Histology Images from {dataset}', fontsize='x-large')
    fig.tight_layout()

    fig.savefig(path, dpi=300)
    plt.close()

def plot_exp_frac(param_dict: dict, dataset: str, raw_adata: ad.AnnData, path: str) -> None:
    """ Plot heatmap of expression fraction

    This function plots a heatmap of the expression fraction and global expression fraction for the complete collection of slides.

    Args:
        raw_adata (ad.AnnData): An unfiltered and unprocessed (in raw counts) slide collection.
        path (str): Path to save the plot.
    """

    
    # Find indexes of cells with total_counts outside the range [cell_min_counts, cell_max_counts]
    sample_counts = np.squeeze(np.asarray(raw_adata.X.sum(axis=1)))
    bool_valid_samples = (sample_counts > param_dict['cell_min_counts']) & (sample_counts < param_dict['cell_max_counts'])
    valid_samples = raw_adata.obs_names[bool_valid_samples]

    # Subset the raw_adata to keep only the valid samples
    raw_adata = raw_adata[valid_samples, :].copy()

    # Compute the min expression fraction for each gene across all the slides
    raw_adata = gene_features.get_exp_frac(raw_adata)
    # Compute the global expression fraction for each gene
    raw_adata = gene_features.get_glob_exp_frac(raw_adata)
    
    # Histogram matrix
    hist_mat, edge_exp_frac, edge_glob_exp_frac = np.histogram2d(raw_adata.var['exp_frac'], raw_adata.var['glob_exp_frac'], range=[[0,1],[0,1]], bins=20)
    # Define dataframe
    index_str = [f'{int(100*per)}%' for per in edge_exp_frac[1:]]
    col_str = [f'{int(100*per)}%' for per in edge_glob_exp_frac[1:]]

    hist_df = pd.DataFrame(hist_mat.astype(int), index=index_str, columns=col_str)
    # Plot params
    scale = 3
    fig_size = (50, 40)
    tit_size = 80
    lab_size = 40
    
    # Define colormap
    d_colors = ["white", "darkcyan"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", d_colors)

    # Plot global expression fraction dataframe
    plt.figure(figsize=fig_size)
    sns.set_theme(font_scale=scale)
    ax = sns.heatmap(hist_df, annot=True, linewidths=.5, fmt='g', cmap=cmap1, linecolor='k', norm=colors.LogNorm(vmin=0.9, vmax=10000))
    
    # Define figure styling
    plt.suptitle(f'Expression Fraction {dataset}', fontsize=tit_size)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.tick_params(labelsize=lab_size)
    plt.xlabel("Global Expression Fraction", fontsize=tit_size)
    plt.ylabel("Expression Fraction", fontsize=tit_size)

    # Define color bar configs
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lab_size)
    cbar.ax.set_ylabel('Number of Genes', fontsize=tit_size)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(path, dpi=300)
    plt.close()
    # Return font scale to normal and matplotlib defaults to not mess the other figures
    sns.set_theme(font_scale=1.0)
    mpl.rcParams.update(mpl.rcParamsDefault)

def plot_histograms(processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
    """ Plot filtering histograms

    This function plots a figure that analyses the effect of the filtering over the data.
    The first row corresponds to the raw data (which has patches and excludes constant genes) and the second row
    plots the filtered and processed data. Histograms of total:
        
        1. Counts per cell
        2. Cells with expression
        3. Total counts per gene
        4. Moran I statistics (only in processed data)
    
    are generated. The plot is saved in the specified path.
    Cell filtering histograms are in red, gene filtering histograms are in blue and autocorrelation filtering histograms are in green.

    Args:
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        raw_adata (ad.AnnData): Loaded data from .h5ad file that is not filtered but has patch information.
        path (str): Path to save histogram plot.
    """


    # Compute qc metrics for raw and processed data in order to have total counts updated
    sc.pp.calculate_qc_metrics(raw_adata, inplace=True, log1p=False, percent_top=None)
    sc.pp.calculate_qc_metrics(processed_adata, inplace=True, log1p=False, percent_top=None, layer='counts')
    
    # Compute the expression fraction of the raw_adata
    raw_adata = gene_features.get_exp_frac(raw_adata)

    # Create figures
    fig, ax = plt.subplots(nrows=2, ncols=5)
    fig.set_size_inches(18.75, 5)

    bin_num = 50

    # Plot histogram of the number of counts that each cell has
    raw_adata.obs['total_counts'].hist(ax=ax[0,0], bins=bin_num, grid=False, color='k')
    processed_adata.obs['total_counts'].hist(ax=ax[1,0], bins=bin_num, grid=False, color='darkred')

    # Plot histogram of the expression fraction of each gene
    raw_adata.var['exp_frac'].plot(kind='hist', ax=ax[0,1], bins=bin_num, grid=False, color='k', logy=True)
    processed_adata.var['exp_frac'].plot(kind = 'hist', ax=ax[1,1], bins=bin_num, grid=False, color='darkcyan', logy=True)

    # Plot histogram of the number of cells that express a given gene
    raw_adata.var['n_cells_by_counts'].plot(kind='hist', ax=ax[0,2], bins=bin_num, grid=False, color='k', logy=True)
    processed_adata.var['n_cells_by_counts'].plot(kind = 'hist', ax=ax[1,2], bins=bin_num, grid=False, color='darkcyan', logy=True)
    
    # Plot histogram of the number of total counts per gene
    raw_adata.var['total_counts'].plot(kind='hist', ax=ax[0,3], bins=bin_num, grid=False, color='k', logy=True)
    processed_adata.var['total_counts'].plot(kind = 'hist', ax=ax[1,3], bins=bin_num, grid=False, color='darkcyan', logy=True)
    
    # Plot histogram of the MoranI statistic per gene
    # raw_adata.var['moranI'].plot(kind='hist', ax=ax[0,4], bins=bin_num, grid=False, color='k', logy=True)
    processed_adata.var['d_log1p_moran'].plot(kind = 'hist', ax=ax[1,4], bins=bin_num, grid=False, color='darkgreen', logy=True)

    # Lists to format axes
    tit_list = ['Raw: Total counts',        'Raw: Expression fraction',         'Raw: Cells with expression',       'Raw: Total gene counts',       'Raw: MoranI statistic',
                'Processed: Total counts',  'Processed: Expression fraction',   'Processed: Cells with expression', 'Processed: Total gene counts', 'Processed: MoranI statistic']
    x_lab_list = ['Total counts', 'Expression fraction', 'Cells with expression', 'Total counts', 'MoranI statistic']*2
    y_lab_list = ['# of cells', '# of genes', '# of genes', '# of genes', '# of genes']*2

    # Format axes
    for i, axis in enumerate(ax.flatten()):
        # Not show moran in raw data because it has no sense to compute it
        if i == 4:
            # Delete frame 
            axis.axis('off')
            continue
        axis.set_title(tit_list[i])
        axis.set_xlabel(x_lab_list[i])
        axis.set_ylabel(y_lab_list[i])
        axis.spines[['right', 'top']].set_visible(False)

    # Shared x axes between plots
    ax[1,0].sharex(ax[0,0])
    ax[1,1].sharex(ax[0,1])
    ax[1,2].sharex(ax[0,2])
    ax[1,3].sharex(ax[0,3])
    ax[1,4].sharex(ax[0,4])

    # Shared y axes between
    ax[1,0].sharey(ax[0,0])
    
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()
    
def plot_random_patches(dataset: str, processed_adata: ad.AnnData, path: str, patch_size: int = 224) -> None:
    """ Plot random set of patches

    This function gets 16 flat random patches (with the specified dims) from the processed adata objects. It
    reshapes them to a bidimensional form and shows them. The plot is saved to the specified path.

    Args:
        patch_size: Patch size (default 224)
        dataset: Name of the dataset
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        path (str): Path to save the image.
    """

    # Verify that the patch scale exists and only exists once
    obsm_keys = list(processed_adata.obsm.keys())
    patch_scale_key = [key for key in obsm_keys if "patches_scale" in key]
    assert len(patch_scale_key) == 1, "patches_scale key either does not exist or exists more than once in keys_list."
    # Get the patch scale
    patch_scale = patch_scale_key[0].split('_')[-1]
    # Get the flat patches from the dataset
    flat_patches = processed_adata.obsm[f'patches_scale_{patch_scale}']
    # Reshape the patches for them to have image form
    patches = flat_patches.reshape((-1, patch_size, patch_size, 3))
    # Choose 16 random patches
    chosen = np.random.randint(low=0, high=patches.shape[0], size=16)
    # Get plotting patches
    plotting_patches = patches[chosen, :, :, :]

    # Declare image
    im, ax = plt.subplots(nrows=4, ncols=4)

    # Cycle over each random patch
    for i, ax in enumerate(ax.reshape(-1)):
        ax.imshow(plotting_patches[i, :, :, :])
        ax.axis('off')
    
    # Set figure formatting
    im.suptitle(f'Random patches from {dataset}')
    plt.tight_layout()
    im.savefig(path, dpi=300)
    plt.close()

def visualize_moran_filtering(param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str, split_names:dict, top: bool = True) -> None:
    """ Plot the most or least auto-correlated genes

    This function visualizes the spatial expression of the 4 most and least auto-correlated genes in processed_adata.
    The title of each subplot shows the value of the moran I statistic for a given gene. The plot is saved to the specified
    path. This plot uses the slide list in string format in param_dict['plotting_slides'] to plot these specific observations.
    If no list is provided (param_dict['plotting_slides']=='None'), 4 random slides are chosen. 

    Args:
        param_dict: Dictionary with dataset parameters
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model
        from_layer (str): Layer of the adata object to use for plotting
        path (str): Path to save the generated image
        split_names (dict): dictionary containing split names
        top (bool, optional): If True, the top 4 most auto-correlated genes are visualized. If False, the top 4 least
                            auto-correlated genes are visualized. Defaults to True
    """
    
    plotting_key = from_layer
    
    # Refine plotting slides string to assure they are in the dataset
    param_dict['plotting_slides'] = refine_plotting_slides_str(split_names, processed_adata, param_dict['plotting_slides'])

    # Get the slides to visualize in adata format
    s_adata_list = filtering.get_slides_adata(processed_adata, param_dict['plotting_slides'])

    # Get te top 4 most or least auto-correlated genes in processed data depending on the value of top
    # NOTE: The selection of genes is done in the complete collection of slides, not in the specified slides
    moran_key = 'd_log1p_moran'
    if top:
        selected_table = processed_adata.var.nlargest(4, columns=moran_key)
    else:
        selected_table = processed_adata.var.nsmallest(4, columns=moran_key)

    # Declare figure
    fig, ax = plt.subplots(nrows=4, ncols=len(s_adata_list))
    fig.set_size_inches(4 * len(s_adata_list) , 13)

    # Cycle over slides
    for i in range(len(selected_table)):

        # Get min and max of the selected gene in the slides
        gene_min = min([dat[:, selected_table.index[i]].layers[plotting_key].min() for dat in s_adata_list])
        gene_max = max([dat[:, selected_table.index[i]].layers[plotting_key].max() for dat in s_adata_list])

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

        for j in range(len(s_adata_list)):
            
            # Define bool to only plot the colorbar in the last column
            cbar = True if j==(len(s_adata_list)-1) else False

            # Plot selected genes in the specified slides
            sq.pl.spatial_scatter(s_adata_list[j], color=[selected_table.index[i]], layer= plotting_key, ax=ax[i,j], cmap='jet', norm=norm, colorbar=cbar)
            
            # Set slide name
            if i==0:
                ax[i,j].set_title(f'{param_dict["plotting_slides"].split(",")[j]}', fontsize=15)
            else:
                ax[i,j].set_title('')
            
            # Set gene name and moran I value
            if j==0:
                ax[i,j].set_ylabel(f'{selected_table.index[i]}: $I = {selected_table[moran_key].iloc[i].round(3)}$', fontsize=13)
            else:
                ax[i,j].set_ylabel('')

    # Format figure
    for axis in ax.flatten():
        axis.set_xlabel('')
        # Turn off all spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)

    # Define title
    tit_str = 'most (top)' if top else 'least (bottom)'

    fig.suptitle(f'Top 4 {tit_str} auto-correlated genes in processed data', fontsize=20)
    fig.tight_layout()
    # Save plot 
    fig.savefig(path, dpi=300)
    plt.close()

def visualize_gene_expression(param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str, split_names:dict) -> None:
    """ Plot specific gene expression

    This function selects the genes specified in param_dict['plotting_genes'] and param_dict['plotting_slides']
    to plot gene expression for the specified genes in the specified slides. If either of them is 'None', then the method
    chooses randomly (4 genes or 4 slides in the stnet_dataset or 2 slides in visium datasets). The data is plotted from
    the .layers[from_layer] expression matrix

    Args:
        param_dict: Dictionary of dataset parameters
        processed_adata (ad.AnnData): The processed adata with the filtered patient collection
        from_layer (str): The key to the layer of the data to plot
        path (str): Path to save the image
        split_names (dict): dictionary containing split names
    """
    # Refine plotting slides string to assure they are in the dataset
    param_dict['plotting_slides'] = refine_plotting_slides_str(split_names, processed_adata, param_dict['plotting_slides'])

    # Get the slides to visualize in adata format
    s_adata_list = filtering.get_slides_adata(processed_adata, param_dict['plotting_slides'])

    # Define gene list
    gene_list = param_dict['plotting_genes'].split(',')

    # Try to get the specified genes otherwise choose randomly
    try:
        gene_table = processed_adata[:, gene_list].var
    except:
        print('Could not find all the specified plotting genes, choosing randomly')
        gene_list = np.random.choice(processed_adata.var_names, size=4, replace=False)
        gene_table = processed_adata[:, gene_list].var
    
    # Declare figure
    fig, ax = plt.subplots(nrows=4, ncols=len(s_adata_list))
    fig.set_size_inches(4 * len(s_adata_list) , 13)

    # Cycle over slides
    for i in range(len(gene_table)):

        # Get min and max of the selected gene in the slides
        gene_min = min([dat[:, gene_table.index[i]].layers[from_layer].min() for dat in s_adata_list])
        gene_max = max([dat[:, gene_table.index[i]].layers[from_layer].max() for dat in s_adata_list])

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

        for j in range(len(s_adata_list)):

            # Define bool to only plot the colorbar in the last column
            cbar = True if j==(len(s_adata_list)-1) else False
            
            # Plot selected genes in the specified slides
            sq.pl.spatial_scatter(s_adata_list[j], layer=from_layer, color=[gene_table.index[i]], ax=ax[i,j], cmap='jet', norm=norm, colorbar=cbar)
            
            # Set slide name
            if i==0:
                ax[i,j].set_title(f'{param_dict["plotting_slides"].split(",")[j]}', fontsize=15)
            else:
                ax[i,j].set_title('')
            
            # Set gene name with moran I value 
            if j==0:
                moran_key = 'd_log1p_moran'
                ax[i,j].set_ylabel(f'{gene_table.index[i]}: $I = {gene_table[moran_key].iloc[i].round(3)}$', fontsize=13)
            else:
                ax[i,j].set_ylabel('')
    
    # Format figure
    for axis in ax.flatten():
        axis.set_xlabel('')
        # Turn off all spines
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
    
    fig.suptitle('Gene expression in processed data', fontsize=20)
    fig.tight_layout()
    # Save plot
    fig.savefig(path, dpi=300)
    plt.close()

def plot_clusters(dataset: str, param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str, split_names:dict) -> None:
    """ Plot clusters spatially

    This function generates a plot that visualizes Leiden clusters spatially in the slides in param_dict['plotting_slides'].
    The slides can be specified in param_dict['plotting_slides'] or chosen randomly.
    
    It plots:
        1. The spatial distribution of the Leiden clusters in the slides.
        2. UMAP embeddings of each slide colored by Leiden clusters.
        3. General UMAP embedding of the complete dataset colored by Leiden clusters and the batch correction key.
        4. PCA embeddings of the complete dataset colored by the batch correction key.

    Args:
        dataset: Name of the dataset
        param_dict: Dictionary of dataset parameters
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        from_layer (str): The key in adata.layers where the expression matrix is stored.
        path (str): Path to save the image
        split_names (dict): dictionary containing split names
    """

    # Update the adata object with the embeddings and clusters
    updated_adata = spot_features.compute_dim_red(processed_adata, from_layer)

    # Refine plotting slides string to assure they are in the dataset
    param_dict['plotting_slides'] = refine_plotting_slides_str(split_names, processed_adata, param_dict['plotting_slides'])

    # Get the slides to visualize in adata format
    s_adata_list = filtering.get_slides_adata(updated_adata, param_dict['plotting_slides'])

    # Define dictionary from cluster to color
    clusters = updated_adata.obs['cluster'].unique()
    # Sort clusters
    clusters = np.sort([int(cl) for cl in clusters])
    clusters = [str(cl) for cl in clusters]
    # Define color palette
    colors = sns.color_palette('hls', len(clusters))
    palette = dict(zip(clusters, colors))
    gray_palette = dict(zip(clusters, ['gray']*len(clusters)))

    # Declare figure
    fig = plt.figure(layout="constrained")
    gs0 = fig.add_gridspec(1, 2)
    gs00 = gs0[0].subgridspec(4, 2)
    gs01 = gs0[1].subgridspec(3, 1)

    fig.set_size_inches(15,14)

    # Cycle over slides
    for i in range(len(s_adata_list)):
        
        curr_clusters = s_adata_list[i].obs['cluster'].unique()
        # Sort clusters
        curr_clusters = np.sort([int(cl) for cl in curr_clusters])
        curr_clusters = [str(cl) for cl in curr_clusters]
        # # Define color palette
        spatial_colors = matplotlib.colors.ListedColormap([palette[x] for x in curr_clusters])
        
        # Get ax for spatial plot and UMAP plot
        spatial_ax = fig.add_subplot(gs00[i, 0])
        umap_ax = fig.add_subplot(gs00[i, 1])

        # Plot cluster colors in spatial space
        sq.pl.spatial_scatter(s_adata_list[i], color=['cluster'], ax=spatial_ax, palette=spatial_colors)

        spatial_ax.get_legend().remove()
        spatial_ax.set_title('Spatial', fontsize=18)
        spatial_ax.set_ylabel(f'{param_dict["plotting_slides"].split(",")[i]}', fontsize=12)
        spatial_ax.set_xlabel('')
        # Turn off all spines
        spatial_ax.spines['top'].set_visible(False)
        spatial_ax.spines['right'].set_visible(False)
        spatial_ax.spines['bottom'].set_visible(False)
        spatial_ax.spines['left'].set_visible(False)        

        # Plot cluster colors in UMAP space for slide and all collection
        sc.pl.umap(updated_adata, layer=from_layer, color=['cluster'], ax=umap_ax, frameon=False, palette=gray_palette, s=10, cmap=None, alpha=0.2)
        umap_ax.get_legend().remove()
        sc.pl.umap(s_adata_list[i], layer=from_layer, color=['cluster'], ax=umap_ax, frameon=False, palette=palette, s=10, cmap=None)
        umap_ax.get_legend().remove()
        umap_ax.set_title('UMAP', fontsize=18)
        
    # Get axes for leiden clusters, patient and cancer types
    leiden_ax = fig.add_subplot(gs01[0])
    patient_ax = fig.add_subplot(gs01[1])
    pca_ax = fig.add_subplot(gs01[2])

    # Plot leiden clusters in UMAP space
    sc.pl.umap(updated_adata, color=['cluster'], ax=leiden_ax, frameon=False, palette=palette, s=10, cmap=None)
    leiden_ax.get_legend().set_title('Leiden Clusters')
    leiden_ax.get_legend().get_title().set_fontsize(15)
    leiden_ax.set_title('UMAP & Leiden Clusters', fontsize=18)

    # Plot batch_key in UMAP space
    sc.pl.umap(updated_adata, color=[param_dict['combat_key']], ax=patient_ax, frameon=False, palette='tab20', s=10, cmap=None)
    patient_ax.get_legend().set_title(param_dict['combat_key'])
    patient_ax.get_legend().get_title().set_fontsize(15)
    patient_ax.set_title(f"UMAP & {param_dict['combat_key']}", fontsize=18)

    # Plot cancer types in UMAP space
    sc.pl.pca(updated_adata, color=[param_dict['combat_key']], ax=pca_ax, frameon=False, palette='tab20', s=10, cmap=None)
    pca_ax.get_legend().set_title(param_dict['combat_key'])
    pca_ax.get_legend().get_title().set_fontsize(15)
    pca_ax.set_title(f'PCA & {param_dict["combat_key"]}', fontsize=18)
    
    # Format figure and save
    fig.suptitle(f'Cluster visualization for {dataset} in layer {from_layer}', fontsize=25)
    # fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_mean_std(dataset: str, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
    """ Plot mean and std of all genes

    This function plots a scatter of mean and standard deviation of genes present in raw_adata (black) and all the layers with non-zero
    mean in processed_adata. It is used to see the effect of filtering and processing in the genes. The plot is saved to the specified path.

    Args:
        dataset: Name of the dataset
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        raw_adata (ad.AnnData): Data loaded data from .h5ad file that is not filtered but has patch information.
        path (str): Path to save the image.
    """
    # Copy raw data to auxiliary data
    aux_raw_adata = raw_adata.copy()

    # Normalize and log transform aux_raw_adata
    sc.pp.normalize_total(aux_raw_adata, inplace=True)
    sc.pp.log1p(aux_raw_adata)

    # Get means and stds from raw data
    raw_mean = aux_raw_adata.to_df().mean(axis=0)
    raw_std = aux_raw_adata.to_df().std(axis=0)

    # Define list of layers to plot
    layers = ['log1p', 'd_log1p', 'c_log1p', 'c_d_log1p']

    plt.figure()
    plt.scatter(raw_mean, raw_std, s=1, c='k', label='Raw data')
    for layer in layers:
        # Get means and stds from processed data
        pro_mean = processed_adata.to_df(layer=layer).mean(axis=0)
        pro_std = processed_adata.to_df(layer=layer).std(axis=0)
        plt.scatter(pro_mean, pro_std, s=1, label=f'{layer} data')
    plt.xlabel('Mean $Log(x+1)$')
    plt.ylabel('Std $Log(x+1)$')
    plt.legend(loc='best')
    plt.title(f'Mean Std plot {dataset}')
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_data_distribution_stats(dataset: str, processed_adata: ad.AnnData, path:str) -> None:
    """ Plot dataset's general stats

    This function plots a pie chart and bar plots of the distribution of spots and slides in the dataset split.

    Args:
        dataset: Name of the dataset
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        path (str): Path to save the image.
    """
    patients = processed_adata.obs['patient'].unique()
    slides = processed_adata.obs['slide_id'].unique()
    quant_spots = [processed_adata[processed_adata.obs['split'] == split].shape[0] for split in ['train', 'val', 'test']]
    metadata = f"Patients: {len(patients)}\nSlides: {len(slides)}\nSpots: {processed_adata.shape[0]}"
    labels_pie = ['Train', 'Valid', 'Test']
    if quant_spots[-1] == 0:
        quant_spots.pop()
        labels_pie.pop()

    # Create figures
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(18.75, 6.5)
    
    # Format axes
    ax[0].pie(quant_spots, labels = labels_pie,
            wedgeprops = {"linewidth": 1, "edgecolor": "white"}, 
            autopct = lambda x: '{:.1f}%\n{:.0f}'.format(x, x*processed_adata.shape[0]/100), 
            colors = sns.color_palette('Set2'),
            textprops = {"fontsize": 15})

    spot_counts = processed_adata.obs.groupby(['patient', 'split'], observed=False)['unique_id'].nunique().unstack(fill_value=0)
    spot_counts = spot_counts.reindex(columns=['train','val','test'])
    slide_counts = processed_adata.obs.groupby(['patient', 'split'], observed=False)['slide_id'].nunique().unstack(fill_value=0)
    slide_counts = slide_counts.reindex(columns=['train','val','test'])

    spot_counts.plot.bar(stacked=True, rot=0, color=sns.color_palette('Set2'), ax=ax[1], legend=False, fontsize=13)
    slide_counts.plot.bar(stacked=True, rot=0, color=sns.color_palette('Set2'), ax=ax[2], legend=False, fontsize=13)

    ax[1].spines[['right', 'top']].set_visible(False)
    ax[2].spines[['right', 'top']].set_visible(False)

    ax[0].set_title('Spot distribution per split', fontsize = 17)
    ax[1].set_title('Spots per patient', fontsize = 17)
    ax[2].set_title('Slides per patient', fontsize = 17)

    ax[1].set_ylabel('# spots', fontsize = 15)
    ax[2].set_ylabel('# slides', fontsize = 15)

    ax[1].set_xlabel('Patient', fontsize = 15)
    ax[2].set_xlabel('Patient', fontsize = 15)

    fig.suptitle(f"Data distribution for {dataset}", fontsize = 20)
    plt.figtext(0.02, 0.08, metadata, fontsize=15, wrap=True, bbox ={'facecolor':'whitesmoke', 'alpha':0.3, 'pad':5})
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close()

def plot_mean_std_partitions(dataset: str, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
    """ Plot mean and std of genes by data split

    This function plots a scatter of mean and standard deviation of genes present in processed_adata drawing with a different color different data
    splits (train/val/test). This is all done for the specified layer in the from_layer parameter. This function is used to see how tractable is
    the task. The plot is saved to the specified path.

    Args:
        dataset: Name of the dataset
        processed_adata (ad.AnnData): Processed and filtered data ready to use by the model.
        from_layer (str): The key in adata.layers where the expression matrix is stored.
        path (str): Path to save the image.
    """
    
    
    # Copy processed adata to avoid problems
    aux_processed_adata = processed_adata.copy() 

    plt.figure()
    for curr_split in aux_processed_adata.obs['split'].unique():
        # Get means and stds from processed data
        curr_mean = aux_processed_adata[aux_processed_adata.obs.split==curr_split, :].to_df(layer=from_layer).mean(axis=0)
        curr_std = aux_processed_adata[aux_processed_adata.obs.split==curr_split, :].to_df(layer=from_layer).std(axis=0)
        plt.scatter(curr_mean, curr_std, s=1, label=f'{curr_split} data')
    plt.xlabel('Mean $Log(x+1)$')
    plt.ylabel('Std $Log(x+1)$')
    plt.legend(loc='best')
    plt.title(f'Mean Std plot {dataset}')
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

#TODO: revisar para eventualmente eliminar (se usa en plot_test)
def refine_plotting_slides_str(split_names: dict, collection: ad.AnnData, slide_list: str) -> str:
    """ Assure plotting slides are on the dataset.

    This function refines the plotting slides string to assure all slides are on the dataset. It works in the following way:

    1. If all slides are in the dataset it does nothing and returns the same slide_list parameter.
    2. If any slide is missing in the dataset or slide_list=='None' then it does one of 2 things:

        a. If the dataset has 4 or less slides all slides are set as plotting slides
        b. If the dataset has more than 4 slides it iterates over splits (train/val/test) and choses a single slide at a time without replacement. 

    Args:
        split_names: dictionary containing split names
        collection (ad.AnnData): Processed and filtered data ready to use by the model.
        slide_list (str): String with a list of slides separated by commas.

    Returns:
        str: Refined version of the slide_list string.
    """
    # Get bool value indicating if all slides in the string are on the dataset
    plot_slides_in_dataset = all([sl in collection.obs.slide_id.unique() for sl in slide_list.split(',')])

    # Decide if a refinement must be done
    if (not plot_slides_in_dataset) or (slide_list=='None'):
        
        # Check if there are less than 4 slides. If so, all are plotting slides
        if len(collection.obs.slide_id.unique()) <= 4:
            slide_list = ','.join(collection.obs.slide_id.unique())
            print(f'Plotting slides were None or missing in the dataset. And there are 4 or less slides. Setting all slides as plotting slides: {slide_list}')
        
        # If more than 4 slides, iterate over splits and chose randomly without replacement
        else:
            plotting_slide_list = []
            # Get list of unique splits
            split_list = collection.obs.split.unique()
            # Get a copy of the dictionary of splits to slides
            split2slide_list = split_names.copy()
            
            # Set counter to 0
            count = 0

            # Iterate until we get 4 slides
            while len(plotting_slide_list) < 4:
                
                # Get current split ant current slide list
                curr_split = split_list[count % len(split_list)]
                curr_slide_list = split2slide_list[curr_split]

                # If the current slide list has slides choose one and delete it from the list
                if len(curr_slide_list) > 0:
                    curr_slide = np.random.choice(curr_slide_list, 1)[0]
                    plotting_slide_list.append(curr_slide)
                    split2slide_list[curr_split].remove(curr_slide)
                
                # Update counter
                count+=1
            
            # Update dataset parameter
            slide_list = ','.join(plotting_slide_list)
            print(f'Plotting slides were None or missing in the dataset. And there are more than 4 slides. Setting slides internally from all splits: {slide_list}')
    
    return slide_list

def plot_tests(patch_size: int, dataset: str, split_names: dict, param_dict: dict, folder_path: str, processed_adata: ad.AnnData, raw_adata: ad.AnnData)->None:
    """ Plot all quality control plots

    This function calls all the plotting functions in the class to create 6 quality control plots to check if the processing step of
    the dataset is performed correctly. The results are saved in dataset_logs folder and indexed by date and time. A dictionary
    in json format with all the dataset parameters is saved in the same log folder for reproducibility. Finally, a txt with the names of the
    genes used in processed adata is also saved in the folder.
    """

    ### Define function to get an adata list of plotting slides

    print('Started quality control plotting')
    start = time()

    # Define directory path to save data
    save_path = os.path.join(folder_path, 'qc_plots')
    os.makedirs(save_path, exist_ok=True)
    # Define interest layers
    relevant_layers = ['log1p', 'd_log1p', 'c_d_log1p']
    complete_layers = ['counts', 'tpm', 'log1p', 'd_log1p', 'c_d_log1p']
    
    # Assure that the plotting genes are in the data and if not, set random plotting genes
    if not all([gene in processed_adata.var_names for gene in param_dict['plotting_genes'].split(',')]):
        param_dict['plotting_genes'] = ','.join(np.random.choice(processed_adata.var_names, 4))
        print(f'Plotting genes not in data. Setting random plotting genes: {param_dict["plotting_genes"]}')
    
    # Refine plotting slides string to assure they are in the dataset
    param_dict['plotting_slides'] = refine_plotting_slides_str(split_names, processed_adata, param_dict['plotting_slides'])

    # Plot partitions mean vs std scatter
    print('Started partitions mean vs std scatter plotting')
    os.makedirs(os.path.join(save_path, 'mean_vs_std_partitions'), exist_ok=True)
    for lay in tqdm(relevant_layers):
        plot_mean_std_partitions(dataset, processed_adata, from_layer=lay, path=os.path.join(save_path, 'mean_vs_std_partitions', f'{lay}.png'))
 
    # Plot all slides in collection
    print('Started all slides plotting')
    plot_all_slides(dataset, processed_adata, os.path.join(save_path, 'all_slides.png'))
 
    # Make plot of filtering histograms
    print('Started filtering histograms plotting')
    plot_histograms(processed_adata, raw_adata, os.path.join(save_path, 'filtering_histograms.png'))

    # Make plot of random patches
    print('Started random patches plotting')
    plot_random_patches(dataset, processed_adata, os.path.join(save_path, 'random_patches.png'), patch_size)

    # Create save paths fot top and bottom moran genes
    os.makedirs(os.path.join(save_path, 'top_moran_genes'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'bottom_moran_genes'), exist_ok=True)
    print('Started moran filtering plotting')
    # Plot moran filtering
    for lay in tqdm(relevant_layers):
        # Make plot of 4 most moran genes and 4 less moran genes (in the chosen slides)
        visualize_moran_filtering(param_dict, processed_adata, from_layer=lay, path = os.path.join(save_path, 'top_moran_genes', f'{lay}.png'), split_names=split_names, top = True)
        visualize_moran_filtering(param_dict, processed_adata, from_layer=lay, path = os.path.join(save_path, 'bottom_moran_genes', f'{lay}.png'), split_names=split_names, top = False)
    

    # Create save paths for cluster plots
    os.makedirs(os.path.join(save_path, 'cluster_plots'), exist_ok=True)
    print('Started cluster plotting')
    # Plot cluster graphs
    for lay in tqdm(relevant_layers):
        plot_clusters(dataset, param_dict, processed_adata, from_layer=lay, path=os.path.join(save_path, 'cluster_plots', f'{lay}.png'), split_names=split_names,)
    
    # Define expression layers
    os.makedirs(os.path.join(save_path, 'expression_plots'), exist_ok=True)
    print('Started gene expression plotting')
    # Plot of gene expression in the chosen slides for the 4 chosen genes
    for lay in tqdm(complete_layers):
        visualize_gene_expression(param_dict, processed_adata, from_layer=lay, path=os.path.join(save_path,'expression_plots', f'{lay}.png'), split_names=split_names,)

    # Make plot of mean vs std per gene must be programmed manually.
    print('Started mean vs std plotting')
    plot_mean_std(dataset, processed_adata, raw_adata, os.path.join(save_path, 'mean_std_scatter.png'))

    # Make plot of data distribution statistics.
    print('Started data distribution statistics plotting')
    plot_data_distribution_stats(dataset, processed_adata, os.path.join(save_path, 'splits_stats.png'))

    # Plot expression fraction 2D histogram
    print('Started expression fraction plotting')
    plot_exp_frac(param_dict, dataset, raw_adata, os.path.join(save_path, 'exp_frac.png'))
    
    # Print the time that took to plot quality control
    end = time()
    print(f'Quality control plotting took {round(end-start, 2)}s')
    print(f'Images saved in {save_path}')
