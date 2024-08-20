import pathlib
import sys

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import models

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
param_dict = data.param_dict
adata = data.adata
breakpoint()

#class ImageEncoder(torch.nn.Module) --> def __init__(self, backbone, use_pretrained,  latent_dim):
model = models.ImageEncoder(backbone='resnet', use_pretrained=True, latent_dim=adata.n_vars)
#DONE