import pathlib
import sys

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import filtering

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
adata = data.adata
param_dict = data.param_dict
breakpoint()

#def filter_by_moran(adata: ad.AnnData, n_keep: int, from_layer: str) -> ad.AnnData:
adata_moran = filtering.filter_by_moran(adata, n_keep=param_dict['top_moran_genes'], from_layer='d_log1p')
#DONE

#def filter_dataset(adata: ad.AnnData, param_dict: dict) -> ad.AnnData:
adata_filter = filtering.filter_dataset(adata, param_dict)
#DONE

#def get_slide_from_collection(collection: ad.AnnData,  slide: str) -> ad.AnnData:
slide_id = adata.obs.slide_id.unique()[0]
slide_adata = filtering.get_slide_from_collection(collection = adata,  slide=slide_id)
#DONE

#get_slides_adata(collection: ad.AnnData, slide_list: str) -> list:
all_slides = ",".join(adata.obs.slide_id.unique().to_list())
slides_list = filtering.get_slides_adata(collection=adata, slide_list=all_slides)
#DONE