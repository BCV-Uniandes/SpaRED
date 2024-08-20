import pathlib
import sys
import anndata as ad
import os

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import plotting
import spot_features

data = datasets.get_dataset("vicari_mouse_brain", visualize=False)
adata = data.adata
param_dict = data.param_dict

dataset_path = "/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/2024-05-30-13-08-21"
raw_adata = ad.read_h5ad(os.path.join(dataset_path, f'adata_raw.h5ad'))

inv_folder_path="/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/inv_plots"
os.makedirs(inv_folder_path, exist_ok=True)

#def plot_all_slides(dataset: str, processed_adata: ad.AnnData, path: str) -> None:
plotting.plot_all_slides(dataset=data.dataset, processed_adata=adata, path=os.path.join(inv_folder_path, 'all_slides.png'))
#DONE

#def plot_exp_frac(param_dict: dict, dataset: str, raw_adata: ad.AnnData, path: str) -> None:
plotting.plot_exp_frac(param_dict=param_dict, dataset=data.dataset, raw_adata=raw_adata, path=os.path.join(inv_folder_path, 'exp_frac.png'))
#DONE

#def plot_histograms(processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
plotting.plot_histograms(processed_adata=data.adata, raw_adata=raw_adata, path=os.path.join(inv_folder_path, 'filtering_histograms.png'))
#DONE

#def plot_random_patches(dataset: str, processed_adata: ad.AnnData, path: str, patch_scale: float = 1.0, patch_size: int = 224) -> None:
plotting.plot_random_patches(dataset=data.dataset, processed_adata=adata, path=os.path.join(inv_folder_path, 'random_patches.png'), patch_size=data.patch_size)
#DONE

#def visualize_moran_filtering(param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str, top: bool = True) -> None:
os.makedirs(os.path.join(inv_folder_path, 'top_moran_genes'), exist_ok=True)
os.makedirs(os.path.join(inv_folder_path, 'bottom_moran_genes'), exist_ok=True)
layer = 'c_d_log1p'

plotting.visualize_moran_filtering(param_dict=param_dict, processed_adata=adata, from_layer=layer, path=os.path.join(inv_folder_path, 'top_moran_genes', f'{layer}.png'), split_names=data.split_names, top = True)
plotting.visualize_moran_filtering(param_dict=param_dict, processed_adata=adata, from_layer=layer, path = os.path.join(inv_folder_path, 'bottom_moran_genes', f'{layer}.png'), split_names=data.split_names, top = False)
#DONE

#def visualize_gene_expression(param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
os.makedirs(os.path.join(inv_folder_path, 'expression_plots'), exist_ok=True)
layer = 'counts'
plotting.visualize_gene_expression(param_dict=param_dict, processed_adata=adata, from_layer=layer, path=os.path.join(inv_folder_path,'expression_plots', f'{layer}.png'), split_names=data.split_names)
#DONE

#def plot_clusters(dataset: str, param_dict: dict, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
os.makedirs(os.path.join(inv_folder_path, 'cluster_plots'), exist_ok=True)
layer = 'c_d_log1p'
plotting.plot_clusters(dataset=data.dataset, param_dict=param_dict, processed_adata=adata, from_layer=layer, path=os.path.join(inv_folder_path, 'cluster_plots', f'{layer}.png'), split_names=data.split_names)
#DONE

#def plot_mean_std(dataset: str, processed_adata: ad.AnnData, raw_adata: ad.AnnData, path: str) -> None:
plotting.plot_mean_std(dataset=data.dataset, processed_adata=adata, raw_adata=raw_adata, path=os.path.join(inv_folder_path, 'mean_std_scatter.png'))
#DONE

#def plot_data_distribution_stats(dataset: str, processed_adata: ad.AnnData, path:str) -> None:
plotting.plot_data_distribution_stats(dataset=data.dataset, processed_adata=adata, path=os.path.join(inv_folder_path, 'splits_stats.png'))
#DONE

#def plot_mean_std_partitions(dataset: str, processed_adata: ad.AnnData, from_layer: str, path: str) -> None:
os.makedirs(os.path.join(inv_folder_path, 'mean_vs_std_partitions'), exist_ok=True)
layer = 'c_d_log1p'
plotting.plot_mean_std_partitions(dataset=data.dataset, processed_adata=adata, from_layer=layer, path=os.path.join(inv_folder_path, 'mean_vs_std_partitions', f'{layer}.png'))
#DONE

#def plot_tests(patch_scale: float, patch_size: int, dataset: str, split_names: dict, param_dict: dict, folder_path: str, processed_adata: ad.AnnData, raw_adata: ad.AnnData)->None:
folder_path="/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/all_plots"
os.makedirs(folder_path, exist_ok=True)
plotting.plot_tests(patch_size=data.patch_size, dataset=data.dataset, split_names=data.split_names, param_dict=param_dict, folder_path=folder_path, processed_adata=adata, raw_adata=raw_adata)
#DONE
