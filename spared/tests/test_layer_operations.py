import pathlib
import sys
import anndata as ad
import os

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import layer_operations

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
param_dict = data.param_dict

dataset_path = "/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/2024-05-30-13-08-21"
adata = ad.read_h5ad(os.path.join(dataset_path, f'adata_raw.h5ad'))
breakpoint()

#def tpm_normalization(organism: str, adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
adata.layers['counts'] = adata.X.toarray()
adata = layer_operations.tpm_normalization(organism=param_dict["organism"], adata=adata, from_layer="counts", to_layer="tpm")
#DONE

#def log1p_transformation(adata=adata, from_layer: str, to_layer: str) -> ad.AnnData:
adata = layer_operations.log1p_transformation(adata, from_layer='tpm', to_layer='log1p')
#DONE

#def combat_transformation(adata: ad.AnnData, batch_key: str, from_layer: str, to_layer: str) -> ad.AnnData:
adata = layer_operations.combat_transformation(adata, batch_key=param_dict['combat_key'], from_layer='log1p', to_layer='c_log1p')
#DONE

#def get_deltas(adata: ad.AnnData, from_layer: str, to_layer: str) -> ad.AnnData:
adata = layer_operations.get_deltas(adata, from_layer='log1p', to_layer='deltas')
#DONE

#def add_noisy_layer(adata: ad.AnnData, prediction_layer: str) -> ad.AnnData:
#TODO: recheck when function is modified
adata.layers['mask'] = adata.layers['tpm'] != 0
adata = layer_operations.add_noisy_layer(adata=adata, prediction_layer="c_log1p")
#DONE

#def process_dataset(adata: ad.AnnData, param_dict: dict) -> ad.AnnData:
raw_adata = ad.read_h5ad(os.path.join(dataset_path, f'adata_raw.h5ad'))
processed_adata = layer_operations.process_dataset(adata=raw_adata, param_dict=param_dict)
#DONE