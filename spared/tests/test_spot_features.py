import pathlib
import sys

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import spot_features

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
adata = data.adata
param_dict = data.param_dict
breakpoint()

#def compute_patches_embeddings(adata: ad.AnnData, backbone: str ='densenet', model_path:str="None", patch_size: int = 224) -> None:
spot_features.compute_patches_embeddings(adata=adata, backbone='densenet', model_path="None", patch_size= 224)
#DONE 

#def compute_patches_predictions(adata: ad.AnnData, backbone: str ='densenet', model_path:str="None", patch_size: int = 224) -> None:
spot_features.compute_patches_predictions(adata=adata, backbone='densenet', model_path="None", patch_size= 224)
#DONE

#def compute_dim_red(adata: ad.AnnData, from_layer: str) -> ad.AnnData:
adata = spot_features.compute_dim_red(adata=adata, from_layer="c_d_log1p")
#DONE

#def get_spatial_neighbors(adata: ad.AnnData, n_hops: int, hex_geometry: bool) -> dict:
dict_sn = spot_features.get_spatial_neighbors(adata=adata, n_hops=6, hex_geometry=param_dict["hex_geometry"])
#DONE