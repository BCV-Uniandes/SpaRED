import pathlib
import sys
import anndata as ad
import os

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import gene_features

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
adata = data.adata
param_dict = data.param_dict

dataset_path = "/home/dvegaa/spared/spared/processed_data/villacampa_data/villacampa_lung_organoid/2024-05-30-13-08-21"
adata_raw = ad.read_h5ad(os.path.join(dataset_path, f'adata_raw.h5ad'))
breakpoint()

#def get_exp_frac(adata: ad.AnnData) -> ad.AnnData:
adata_exp = gene_features.get_exp_frac(adata_raw)
#DONE

#def get_glob_exp_frac(adata: ad.AnnData) -> ad.AnnData:
adata_glob_exp = gene_features.get_glob_exp_frac(adata_raw)
#DONE 

#def compute_moran(adata: ad.AnnData, from_layer: str, hex_geometry: bool) -> ad.AnnData:
adata_moran = gene_features.compute_moran(adata=adata, from_layer="c_d_log1p", hex_geometry=param_dict["hex_geometry"])
#DONE
