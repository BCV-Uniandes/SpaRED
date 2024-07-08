import numpy as np
import torch
import warnings
from typing import Union, Tuple
import warnings
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from time import time

warnings.filterwarnings(action='ignore', category=UserWarning)

def pearsonr_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> Tuple[float, list]:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average Pearson correlation.
    To do that, it takes the i-th column of each matrix and computes the Pearson correlation between them.
    It finally returns the average of all the Pearson correlations computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).
    
    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
        detalied_pcc (list): List of pcc for each one of the columns
    """
    masked_gt_mat = torch.masked.masked_tensor(gt_mat, mask=mask)
    masked_gt_mean = masked_gt_mat.mean(dim=0, keepdim=True)

    masked_pred_mat = torch.masked.masked_tensor(pred_mat, mask=mask)
    masked_pred_mean = masked_pred_mat.mean(dim=0, keepdim=True)

    # Construct matrices with only masked means
    masked_gt_mean = masked_gt_mean.to_tensor(float('nan')).repeat(gt_mat.shape[0],1)
    masked_pred_mean = masked_pred_mean.to_tensor(float('nan')).repeat(pred_mat.shape[0],1)

    # Find if there are any columns completely masked
    nan_columns = torch.isnan(masked_gt_mean).all(dim=0)

    # Modify mask==False entries of gt_mat and pred_mat to the masked mean. 
    # NOTE: This replace will make the computation of the metric efficient without taking into account the discarded values of the mask
    gt_mat = torch.where(mask==True, gt_mat, masked_gt_mean)
    pred_mat = torch.where(mask==True, pred_mat, masked_pred_mean)

    # Center both matrices by subtracting the mean of each column
    centered_gt_mat = gt_mat - masked_gt_mean
    centered_pred_mat = pred_mat - masked_pred_mean

    # Remove columns that are completely masked
    centered_gt_mat = centered_gt_mat[:, ~nan_columns]
    centered_pred_mat = centered_pred_mat[:, ~nan_columns]

    # Compute pearson correlation with cosine similarity
    pcc = torch.nn.functional.cosine_similarity(centered_gt_mat, centered_pred_mat, dim=0)

    # Compute mean pearson correlation (the nan mean is to ensure metric computation even when a complete patch is masked)
    mean_pcc = pcc.nanmean().item()
    # Get the list of pccs
    detailed_pcc = pcc.tolist()
    
    return mean_pcc, detailed_pcc

def pearsonr_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> Tuple[float, list]:
    """
    This function uses pearsonr_cols to compute the Pearson correlation between the ground truth and predicted matrices along
    the gene dimension. It is computing the correlation between the true and predicted values for each gene and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each gene.
        detalied_pcc (list): List of pcc for each one of the genes
    """

    mean_pcc, detalied_pcc = pearsonr_cols(gt_mat=gt_mat, pred_mat=pred_mat, mask=mask)

    return mean_pcc, detalied_pcc

def pearsonr_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> Tuple[float, list]:
    """
    This function uses pearsonr_cols to compute the Pearson correlation between the ground truth and predicted matrices along
    the patch dimension. It is computing the correlation the between true and predicted values for each patch and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
        detalied_pcc (list): List of pcc for each one of the patches
    """
    # Transpose matrices and apply pearsonr_torch_cols 
    mean_pcc, detalied_pcc = pearsonr_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T, mask=mask.T)

    return mean_pcc, detalied_pcc

def r2_score_cols(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> Tuple[float, list]:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average R2 score.
    To do that, it takes the i-th column of each matrix and computes the R2 score between them.
    It finally returns the average of all the R2 scores computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).

    Returns:
        mean_r2_score (float): Mean R2 score computed by averaging the R2 score for each column in the matrices.
        detalied_r2_score (list): List of r2 scores for each one of the columns
    """
    # Pass input matrices to masked tensors
    gt_mat = torch.masked.masked_tensor(gt_mat, mask=mask)
    pred_mat = torch.masked.masked_tensor(pred_mat, mask=mask)

    # Remove columns with a single value without masking (these columns make R2 go to infinity)
    single_value_columns =  mask.sum(axis=0)==1
    gt_mat = gt_mat[:, ~single_value_columns]
    pred_mat = pred_mat[:, ~single_value_columns]

    # Compute the column means of the ground truth
    gt_col_means = gt_mat.mean(dim=0).to_tensor(float('nan'))

    # Find if there are any columns completely masked
    nan_columns = torch.isnan(gt_col_means)
    
    # Compute the total sum of squares
    total_sum_squares = torch.sum(torch.square(gt_mat - gt_col_means), dim=0).to_tensor(float('nan'))

    # Compute the residual sum of squares
    residual_sum_squares = torch.sum(torch.square(gt_mat - pred_mat), dim=0).to_tensor(float('nan'))

    # Remove columns that are completely masked
    total_sum_squares = total_sum_squares[~nan_columns]
    residual_sum_squares = residual_sum_squares[~nan_columns]

    # Compute the R2 score for each column
    r2_scores = 1. - (residual_sum_squares / total_sum_squares)

    # Compute the mean R2 score
    mean_r2_score = r2_scores.mean().item()
    detalied_r2_score = r2_scores.tolist()

    return mean_r2_score, detalied_r2_score

def r2_score_gene(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> Tuple[float, list]:
    """
    This function uses r2_score_cols to compute the R2 score between the ground truth and predicted matrices along
    the gene dimension. It is computing the R2 score between the true and predicted values for each gene and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        mean_r2_score (float): Mean R2 score computed by averaging the R2 score for each gene.
        detalied_r2_score (list): List of r2 scores for each one of the genes
    """

    mean_r2_score, detalied_r2_score = r2_score_cols(gt_mat=gt_mat, pred_mat=pred_mat, mask=mask)

    return mean_r2_score, detalied_r2_score

def r2_score_patch(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor) -> Tuple[float, list]:
    """
    This function uses r2_score_cols to compute the R2 score between the ground truth and predicted matrices along
    the patch dimension. It is computing the R2 score between the true and predicted values for each patch and returning the average of all.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_samples, n_genes).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).

    Returns:
        mean_r2_score (float): Mean R2 score computed by averaging the R2 score for each patch.
        detalied_r2_score (list): List of r2 scores for each one of the patches.
    """
    
    # Transpose matrices and apply r2_score_torch_cols
    mean_r2_score, detalied_r2_score = r2_score_cols(gt_mat=gt_mat.T, pred_mat=pred_mat.T, mask=mask.T)
    
    return mean_r2_score, detalied_r2_score

def get_pearsonr(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor, axis:int) -> Tuple[float, list]:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average Pearson correlation.
    To do that, it takes the i-th column of each matrix and computes the Pearson correlation between them.
    It finally returns the average of all the Pearson correlations computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).
        axis (int): wether to compute the pcc by columns (axis=0) ir by rows (axis=1)
    Returns:
        mean_pcc (float): Mean Pearson correlation computed by averaging the Pearson correlation for each patch.
        detalied_pcc (list): List of pcc for each one of the columns
    """
    masked_gt_mat = torch.masked.masked_tensor(gt_mat, mask=mask)
    masked_gt_mean = masked_gt_mat.mean(dim=axis, keepdim=True)

    masked_pred_mat = torch.masked.masked_tensor(pred_mat, mask=mask)
    masked_pred_mean = masked_pred_mat.mean(dim=axis, keepdim=True)

    # Construct matrices with only masked means
    # By columns
    if axis == 0:
        masked_gt_mean = masked_gt_mean.to_tensor(float('nan')).repeat(gt_mat.shape[0],1)
        masked_pred_mean = masked_pred_mean.to_tensor(float('nan')).repeat(pred_mat.shape[0],1)
    # By rows
    elif axis == 1:
        masked_gt_mean = masked_gt_mean.to_tensor(float('nan')).repeat(1,gt_mat.shape[1])
        masked_pred_mean = masked_pred_mean.to_tensor(float('nan')).repeat(1,pred_mat.shape[1])      
    
    # Find if there are any columns completely masked
    nan_axis = torch.isnan(masked_gt_mean).all(dim=axis)

    # Modify mask==False entries of gt_mat and pred_mat to the masked mean. 
    # NOTE: This replace will make the computation of the metric efficient without taking into account the discarded values of the mask
    gt_mat = torch.where(mask==True, gt_mat, masked_gt_mean)
    pred_mat = torch.where(mask==True, pred_mat, masked_pred_mean)

    # Center both matrices by subtracting the mean of each column
    centered_gt_mat = gt_mat - masked_gt_mean
    centered_pred_mat = pred_mat - masked_pred_mean

    # Remove columns that are completely masked
    if axis == 0:  
        centered_gt_mat = centered_gt_mat[:, ~nan_axis]
        centered_pred_mat = centered_pred_mat[:, ~nan_axis]
    # Remove rows that are completely masked
    if axis == 1:  
        centered_gt_mat = centered_gt_mat[~nan_axis, :]
        centered_pred_mat = centered_pred_mat[~nan_axis, :]

    # Compute pearson correlation with cosine similarity
    pcc = torch.nn.functional.cosine_similarity(centered_gt_mat, centered_pred_mat, dim=axis)

    # Compute mean pearson correlation (the nan mean is to ensure metric computation even when a complete patch is masked)
    mean_pcc = pcc.nanmean().item()
    # Get the list of pccs
    detailed_pcc = pcc.tolist()
    
    return mean_pcc, detailed_pcc

def get_r2_score(gt_mat: torch.Tensor, pred_mat: torch.Tensor, mask: torch.Tensor, axis=int) -> Tuple[float, list]:
    """
    This function receives 2 matrices of shapes (n_observations, n_variables) and computes the average R2 score.
    To do that, it takes the i-th column of each matrix and computes the R2 score between them.
    It finally returns the average of all the R2 scores computed.

    Args:
        gt_mat (torch.Tensor): Ground truth matrix of shape (n_observations, n_variables).
        pred_mat (torch.Tensor): Predicted matrix of shape (n_observations, n_variables).
        mask (torch.Tensor): Boolean mask with False in positions that must be ignored in metric computation (n_observations, n_variables).
        axis (int): wether to compute the pcc by columns (axis=0) ir by rows (axis=1)
    Returns:
        mean_r2_score (float): Mean R2 score computed by averaging the R2 score for each column in the matrices.
        detalied_r2_score (list): List of r2 scores for each one of the columns
    """
    # Pass input matrices to masked tensors
    gt_mat = torch.masked.masked_tensor(gt_mat, mask=mask)
    pred_mat = torch.masked.masked_tensor(pred_mat, mask=mask)

    single_value_axis =  mask.sum(axis=axis)==1
    # Remove columns with a single value without masking (these columns make R2 go to infinity)
    if axis == 0:
        gt_mat = gt_mat[:, ~single_value_axis]
        pred_mat = pred_mat[:, ~single_value_axis]
    # Remove rows with a single value without masking (these columns make R2 go to infinity)
    elif axis == 1:
        gt_mat = gt_mat[~single_value_axis, :]
        pred_mat = pred_mat[~single_value_axis, :]
    
    # Compute the axis means of the ground truth
    gt_axis_means = gt_mat.mean(dim=axis, keepdim=True).to_tensor(float('nan'))

    # Find if there are any columns or rows completely masked
    nan_axis = torch.isnan(gt_axis_means).squeeze(dim=axis)
    
    # Compute the total sum of squares
    total_sum_squares = torch.sum(torch.square(gt_mat - gt_axis_means), dim=axis).to_tensor(float('nan'))

    # Compute the residual sum of squares
    residual_sum_squares = torch.sum(torch.square(gt_mat - pred_mat), dim=axis).to_tensor(float('nan'))

    # Remove rows or columns that are completely masked
    total_sum_squares = total_sum_squares[~nan_axis]
    residual_sum_squares = residual_sum_squares[~nan_axis]

    # Compute the R2 score for each row or column
    r2_scores = 1. - (residual_sum_squares / total_sum_squares)

    # Compute the mean R2 score
    mean_r2_score = r2_scores.mean().item()
    detalied_r2_score = r2_scores.tolist()

    return mean_r2_score, detalied_r2_score

def get_metrics(gt_mat: Union[np.array, torch.Tensor], pred_mat: Union[np.array, torch.Tensor], mask: Union[np.array, torch.Tensor], detailed: bool = False) -> dict:
    """ Get general regression metrics

    This function receives 2 matrices of shapes (n_samples, n_genes) and computes the following metrics:
    
        - Pearson correlation (gene-wise) [PCC-Gene]
        - Pearson correlation (patch-wise) [PCC-Patch]
        - r2 score (gene-wise) [R2-Gene]
        - r2 score (patch-wise) [R2-Patch]
        - Mean squared error [MSE]
        - Mean absolute error [MAE]
        - Global metric [Global] (Global = PCC-Gene + R2-Gene + PCC-Patch + R2-Patch - MAE - MSE)
    
    If detailed == True. Then the function returns these aditional keys (all of them are numpy arrays):
        
        - Individual pearson correlation for every gene [PPC-Gene-detailed]
        - Individual pearson correlation for every patch [PPC-Patch-detailed]
        - Individual r2 score for every gene [R2-Gene-detailed]
        - Individual r2 score for every patch [R2-Gene-detailed]
        - Individual MSE for every gene [detailed_mse_gene]
        - Individual MAE for every gene [detailed_mae_gene]
        - Individual average error for every gene [detailed_error_gene]
        - Flat concatenation of all errors in valid positions [detailed_errors]

    Args:
        gt_mat (Union[np.array, torch.Tensor]): Ground truth matrix of shape (n_samples, n_genes).
        pred_mat (Union[np.array, torch.Tensor]): Predicted matrix of shape (n_samples, n_genes).
        mask (Union[np.array, torch.Tensor]): Boolean mask with False in positions that must be ignored in metric computation (n_samples, n_genes).
        detailed (bool): If True, the dictionary also includes the detailed metrics.

    Returns:
        dict: Dictionary containing the metrics computed. The keys are: ['PCC-Gene', 'PCC-Patch', 'R2-Gene', 'R2-Patch', 'MSE', 'MAE', 'Global']
    """

    # Assert that all matrices have the same shape
    assert gt_mat.shape == pred_mat.shape, "gt_mat and pred_mat matrices must have the same shape."
    assert gt_mat.shape == mask.shape, "gt_mat and mask matrices must have the same shape."

    # If input are numpy arrays, convert them to torch tensors
    if isinstance(gt_mat, np.ndarray):
        gt_mat = torch.from_numpy(gt_mat)
    if isinstance(pred_mat, np.ndarray):
        pred_mat = torch.from_numpy(pred_mat)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    # Get boolean indicating constant columns in predicted matrix 
    # NOTE: A constant gene prediction will mess with the pearson correlation
    constant_cols = torch.all(pred_mat == pred_mat[[0],:], axis = 0)
    # Get boolean indicating if there are any constant columns
    any_constant_cols = torch.any(constant_cols)

    # Get boolean indicating constant rows in predicted matrix
    # NOTE: A constant patch prediction will mess with the pearson correlation
    constant_rows = torch.all(pred_mat == pred_mat[:,[0]], axis = 1)
    # Get boolean indicating if there are any constant rows
    any_constant_rows = torch.any(constant_rows)
    # If there are any constant columns, set the pcc_g and r2_g to None
    if any_constant_cols:
        pcc_g, detailed_pcc_g = np.nan, np.nan
        print(f"There are {constant_cols.sum().item()} constant columns in the predicted matrix. This means a gene is being predicted as constant. The Pearson correlation (gene-wise) will be set to NaN.")
    else:
        # Compute Pearson correlation (gene-wise)
        pcc_g, detailed_pcc_g = pearsonr_gene(gt_mat, pred_mat, mask=mask)
    
    # If there are any constant rows, set the pcc_p and r2_p to None
    if any_constant_rows:
        pcc_p, detailed_pcc_p = np.nan, np.nan
        print(f"There are {constant_rows.sum().item()} constant rows in the predicted matrix. This means a patch is being predicted as constant. The Pearson correlation (patch-wise) will be set to NaN.")
    else:
        # Compute Pearson correlation (patch-wise)
        pcc_p, detailed_pcc_p = pearsonr_patch(gt_mat, pred_mat, mask=mask)
        

    # Compute r2 score (gene-wise)
    r2_g, detailed_r2_g = r2_score_gene(gt_mat, pred_mat, mask=mask)
    # Compute r2 score (patch-wise)
    r2_p, detailed_r2_p = r2_score_patch(gt_mat, pred_mat, mask=mask)

    # Compute mean squared error
    mse = torch.nn.functional.mse_loss(gt_mat[mask], pred_mat[mask], reduction='mean').item()
    # Compute mean absolute error
    mae = torch.nn.functional.l1_loss(gt_mat[mask], pred_mat[mask], reduction='mean').item()

    # Compute detailed error metrics (only at gene level because patch level usualy gives empty patches)
    errors = pred_mat - gt_mat
    errors[~mask] = torch.nan
    sq_errors = torch.square(errors)
    detailed_mse_gene = sq_errors.nanmean(dim=0).tolist()
    detailed_mae_gene = torch.abs(errors).nanmean(dim=0).tolist()
    detailed_error_gene = errors.nanmean(dim=0).tolist()
    detailed_errors = errors[mask].tolist()
    
    # Create dictionary with the metrics computed
    metrics_dict = {
        'PCC-Gene': pcc_g,
        'PCC-Patch': pcc_p,
        'R2-Gene': r2_g,
        'R2-Patch': r2_p,
        'MSE': mse,
        'MAE': mae,
        'Global': pcc_g + pcc_p + r2_g + r2_p - mse - mae 
    }
    
    # If detailed metrics are desired then add to the metric dict the PCCs and R2s for every gene and patch
    if detailed==True:
        detailed_metrics_dict = {
            'detailed_PCC-Gene': detailed_pcc_g,
            'detailed_PCC-Patch': detailed_pcc_p,
            'detailed_R2-Gene': detailed_r2_g,
            'detailed_R2-Patch': detailed_r2_p,
            'detailed_mse_gene': detailed_mse_gene,
            'detailed_mae_gene': detailed_mae_gene,
            'detailed_error_gene': detailed_error_gene, 
            'detailed_errors': detailed_errors,
        }
        # Update metric dict
        metrics_dict = {**metrics_dict, **detailed_metrics_dict}

    return metrics_dict

# Here we have some testing code
if __name__=='__main__':
    
    # Set number of observations and genes (hypothetical)
    obs = 7777
    genes = 256
    imputed_fraction = 0.26 # This is the percentage of zeros in the mask

    # Henerate random matrices
    pred = torch.randn((obs,genes))
    gt = torch.randn((obs,genes))
    mask = torch.rand((obs,genes))>imputed_fraction

    # Compute metrics with the fast way (efficient implementation)
    print('Fast metrics'+'-'*40)
    start = time()
    test_metrics = get_metrics(gt, pred, mask=mask)
    print("Time taken: {:5.2f}s".format(time()-start))

    for key, val in test_metrics.items():
        print("{} = {:5.7f}".format(key, val))


    # Compute metrics with the slow way (inefficient implementation but secure)
    print('Slow metrics'+'-'*40) 
    start = time()
    slow_test_metrics = slow_get_metrics(gt, pred, mask=mask)
    print("Time taken: {:5.2f}s".format(time()-start))

    for key, val in slow_test_metrics.items():
        print("{} = {:5.7f}".format(key, val))
