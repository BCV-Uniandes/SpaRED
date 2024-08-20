import torch
import squidpy as sq
from spackle.utils import *

class ImputationDataset(torch.utils.data.Dataset):
    def __init__(self, adata, args_dict, split_name, prediction_layer, pre_masked = False):
        """
        This class prepares the `adata` for processing with SpaCKLE.

        Args:
            adata (ad.AnnData): An AnnData object containing the entire dataset.
            args_dict (dict): A dictionary with the values needed for data processing. For more information on the required keys, refer to the 
                              documentation of the function `get_args_dict()` in `spared.spackle.utils`.
            split_name (str): The name of the data split being processed. This is useful for identifying which data split the model is being tested on.
            prediction_layer (str): The name of the layer that contains the gene expression of the spots. This layer will be randomly masked and predicted on 
                                    to train the model, or it has missing data that will be completed with SpaCKLE.
            pre_masked (str, optional): Specifies if the incoming data has already been masked for testing purposes. 
                * If True, `__getitem__()` will return the random mask that was used to mask the original expression 
                values instead of the median imputation mask, as well as the ground truth expressions and the masked data.
        """


        self.adata = adata
        self.pred_layer = prediction_layer
        self.split_name = split_name
        self.pre_masked = pre_masked
        self.use_visual_features = args_dict["use_visual_features"]
        self.img_enc_backbone = args_dict["img_backbone"]
        # Get original expression matrix based on selected prediction layer.
        self.expression_mtx = torch.tensor(self.adata.layers[self.pred_layer])
        # Retrieve the mask from the adata, where "False" corresponds to the values that contain the median as expression value.
        self.median_imp_mask = torch.tensor(self.adata.layers['mask'])

        # Get the masked expression matrix expression and random mask if data has been pre-masked
        self.pre_masked_expression_mtx = torch.tensor(self.adata.layers['masked_expression_matrix']) if pre_masked else None
        self.random_mask = torch.tensor(self.adata.layers['random_mask']) if pre_masked else None

        # Get adjacency matrix.
        self.adj_mat = None
        self.get_adjacency(args_dict["num_neighs"])

        # Get the embeddings according to the image backbone
        self.embeddings = torch.tensor(self.adata.obsm[f'embeddings_{self.img_enc_backbone}']) if self.use_visual_features else None   

    def get_adjacency(self, num_neighs = 6):
        """
        This function creates the adjacency matrix that indicates which are the num_neighs closest spots to each spot in the slide.
        """
        # Get num_neighs nearest neighbors for each spot
        sq.gr.spatial_neighbors(self.adata, coord_type='generic', n_neighs=num_neighs)
        self.adj_mat = torch.tensor(self.adata.obsp['spatial_connectivities'].todense())

    def build_neighborhood_from_distance(self, idx):
        """
        This function gets the closest n neighbors of the spot in index idx and returns the final neighborhood gene expression matrix,
        as well as the mask that indicates which elements are missing in the original data. If the datasets has already been randomly 
        masked, it will also return the corresponding matrix.
        """
        # Get gt expression for idx spot and its nn
        spot_exp = self.expression_mtx[idx].unsqueeze(dim=0)
        nn_exp = self.expression_mtx[self.adj_mat[:,idx]==1.]
        exp_matrix = torch.cat((spot_exp, nn_exp), dim=0).type('torch.FloatTensor') # Original dtype was 'torch.float64'
        embeddings = None

        # Get image embeddings if needed
        if self.use_visual_features:
            # Get embeddings for idx spot and its nn
            spot_embeddings = self.embeddings[idx].unsqueeze(dim=0)
            nn_embeddings = self.embeddings[self.adj_mat[:,idx]==1.]
            embeddings = torch.cat((spot_embeddings, nn_embeddings), dim=0) 

        if not self.pre_masked:
            # Get median imputation mask for idx spot and its nn
            spot_mask = self.median_imp_mask[idx].unsqueeze(dim=0) #size 1x128
            nn_mask = self.median_imp_mask[self.adj_mat[:,idx]==1.] #size 6 x 128
            median_imp_mask = torch.cat((spot_mask, nn_mask), dim=0)
            # Organize return tuple
            neigh_data = (exp_matrix, median_imp_mask, embeddings)

        else:
            # Get pre-masked expression for idx spot and its nn
            spot_pre_masked_exp = self.pre_masked_expression_mtx[idx].unsqueeze(dim=0) #size 1x128
            nn_pre_masked_exp = self.pre_masked_expression_mtx[self.adj_mat[:,idx]==1.] #size 6 x 128
            pre_masked_exp = torch.cat((spot_pre_masked_exp, nn_pre_masked_exp), dim=0).type('torch.FloatTensor')
            # Get random mask for idx spot and its nn
            spot_random_mask = self.random_mask[idx].unsqueeze(dim=0) #size 1x128
            nn_random_mask = self.random_mask[self.adj_mat[:,idx]==1.] #size 6 x 128
            random_mask = torch.cat((spot_random_mask, nn_random_mask), dim=0)
            # Organize return tuple
            neigh_data = (exp_matrix, pre_masked_exp, random_mask, embeddings)

        return neigh_data

    def __getitem__(self, idx):
        item = {'split_name': self.split_name}

        if not self.pre_masked:
            item['exp_matrix_gt'], item['real_missing'], item['visual_features'] = self.build_neighborhood_from_distance(idx)

        else:
            item['exp_matrix_gt'], item['pre_masked_exp'], item['random_mask'], item['visual_features'] = self.build_neighborhood_from_distance(idx)
                
        if not self.use_visual_features:
            del item['visual_features']

        return item

    def __len__(self):
        return len(self.adata)