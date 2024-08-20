import pathlib
import sys

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import graph_operations
import filtering
import spot_features

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
adata = data.adata
param_dict = data.param_dict
breakpoint()

#Graph operations must have embedding and prediction layers
spot_features.compute_patches_embeddings(adata=adata, backbone='densenet', model_path="None", patch_size= 224)
spot_features.compute_patches_predictions(adata=adata, backbone='densenet', model_path="None", patch_size= 224)

#def get_graphs_one_slide(adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool) -> Tuple[dict,int]:
slide_id = adata.obs.slide_id.unique()[0]
slide_adata = filtering.get_slide_from_collection(adata, slide_id)
dict_graph_slide, max_pos = graph_operations.get_graphs_one_slide(adata=slide_adata, n_hops=6, layer="c_d_log1p", hex_geometry=param_dict["hex_geometry"])
#DONE

#def get_sin_cos_positional_embeddings(graph_dict: dict, max_d_pos: int) -> dict:
dict_pos_emb = graph_operations.get_sin_cos_positional_embeddings(graph_dict=dict_graph_slide, max_d_pos=max_pos)
#DONE

#def get_graphs(adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool=True) -> dict:
dict_graphs = graph_operations.get_graphs(adata=adata, n_hops=6, layer="c_d_log1p", hex_geometry=param_dict["hex_geometry"])
#DONE