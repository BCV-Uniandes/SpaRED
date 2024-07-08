import pathlib
import sys
import torch

SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(SPARED_PATH))
import datasets
import metrics

data = datasets.get_dataset("villacampa_lung_organoid", visualize=False)
adata = data.adata
param_dict = data.param_dict

# Set number of observations and genes (hypothetical)
obs = 10
genes = 8
imputed_fraction = 0.26 # This is the percentage of zeros in the mask

# Henerate random matrices
pred = torch.randn((obs,genes))
gt = torch.randn((obs,genes))
mask = torch.rand((obs,genes))>imputed_fraction

#def get_pearsonr(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor, axis:int) -> Tuple[float, list]:
mean_pcc, list_pcc = metrics.get_pearsonr(gt_mat=gt, pred_mat=pred, mask=mask, axis=0)
mean_pcc, list_pcc = metrics.get_pearsonr(gt_mat=gt, pred_mat=pred, mask=mask, axis=1)
#DONE

#def get_r2_score(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor, axis:int) -> Tuple[float, list]:
mean_pcc, list_pcc = metrics.get_r2_score(gt_mat=gt, pred_mat=pred, mask=mask, axis=0)
mean_pcc, list_pcc = metrics.get_r2_score(gt_mat=gt, pred_mat=pred, mask=mask, axis=1)
#DONE

#def get_metrics(gt_mat: Union[np.array, torch.Tensor], pred_mat: Union[np.array, torch.Tensor], mask: Union[np.array, torch.Tensor], detailed: bool = False) -> dict:
dict_metrics = metrics.get_metrics(gt_mat = gt, pred_mat = pred, mask = mask, detailed = False)
dict_metrics_detailed = metrics.get_metrics(gt_mat = gt, pred_mat = pred, mask = mask, detailed = True)
#DONE