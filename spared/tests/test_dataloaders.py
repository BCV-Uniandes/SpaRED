import pathlib
import sys
import os

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
from gene_features import gene_features
import datasets
import dataloaders

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
param_dict = data.param_dict
adata = data.adata
breakpoint()

#def get_pretrain_dataloaders(adata: ad.AnnData, layer: str = 'c_d_log1p', batch_size: int = 128, shuffle: bool = True, use_cuda: bool = False) -> Tuple[AnnLoader, AnnLoader, AnnLoader]:
train_loader, val_loader, test_loader = dataloaders.get_pretrain_dataloaders(adata = adata, layer = 'c_d_log1p', batch_size = 128, shuffle = True, use_cuda = False)
#DONE

#def get_graph_dataloaders(adata: ad.AnnData, dataset_path: str='', layer: str = 'c_t_log1p', n_hops: int = 2, backbone: str ='densenet', model_path: str = "None", batch_size: int = 128, shuffle: bool = True, hex_geometry: bool=True, patch_size: int=224) -> Tuple[geo_DataLoader, geo_DataLoader, geo_DataLoader]:
graphs_path="/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/graphs"
os.makedirs(graphs_path, exist_ok=True)
train_graph_loader, val_graph_loader, test_graph_loader = dataloaders.get_graph_dataloaders(adata = adata, dataset_path = graphs_path, layer = 'c_d_log1p', n_hops = 2, backbone = 'densenet', model_path = "None", batch_size = 128, shuffle = True, hex_geometry = param_dict["hex_geometry"], patch_size = 224)
#DONE