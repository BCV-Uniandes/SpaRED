import torch
import squidpy as sq
from spackle.utils import *

class ImputationDataset(torch.utils.data.Dataset):
    def __init__(self, adata, args, split_name, prediction_layer, pre_masked = False):
        # TODO: update this description (this class doesn't download the data)
        """
        This is a spatial data class that contains all the information about the dataset. It will call a reader class depending on the type
        of dataset (by now only visium and STNet are supported). The reader class will download the data and read it into an AnnData collection
        object. Then the dataset class will filter, process and plot quality control graphs for the dataset. The processed dataset will be stored
        for rapid access in the future.

        Args:
            adata (ad.AnnData): An anndata object with the data of the entire dataset.
            args (argparse): parser with the values necessary for data processing.
            split_name (str): name of the data split being processed. Useful for identifying which data split the model is being tested on.
            pre_masked (str, optional): specifies if the data incoming has already been masked for testing purposes. 
                    * If True, __getitem__() will return the random mask that was used to mask the original expression 
                    values instead of the median imputation mask, as well as the gt expressions and the masked data.
        """

        self.adata = adata
        self.pred_layer = prediction_layer
        self.split_name = split_name
        self.pre_masked = pre_masked
        self.use_visual_features = args.use_visual_features
        self.img_enc_backbone = args.img_backbone
        # Get original expression matrix based on selected prediction layer.
        self.expression_mtx = torch.tensor(self.adata.layers[self.pred_layer])
        # Retreive the mask from the adata, where "False" corresponds to the values that contain the median as expression value.
        self.median_imp_mask = torch.tensor(self.adata.layers['mask'])

        # Get the masked expression matrix expression and random mask if data has been pre-masked
        self.pre_masked_expression_mtx = torch.tensor(self.adata.layers['masked_expression_matrix']) if pre_masked else None
        self.random_mask = torch.tensor(self.adata.layers['random_mask']) if pre_masked else None

        # Choose method for building neighborhood. Either based on hops or on distance.
        self.neighborhood_type = args.neighborhood_type

        if self.neighborhood_type == 'circular_hops':
            # Get dictionary of neighbors for a given number of hops
            self.neighbors_dict = get_spatial_neighbors(self.adata, args.num_hops, hex_geometry = args.hex_geometry)

        else: # When self.neighborhood_type == 'nn_distance'
            # Get adjacency matrix.
            self.adj_mat = None
            self.get_adjacency(args.num_neighs)

        # Get the embeddings according to the image backbone
        self.embeddings = torch.tensor(self.adata.obsm[f'embeddings_{self.img_enc_backbone}']) if self.use_visual_features else None   

    def get_adjacency(self, num_neighs = 6):
        """
        Function description
        """
        # Get num_neighs nearest neighbors for each spot
        sq.gr.spatial_neighbors(self.adata, coord_type='generic', n_neighs=num_neighs)
        self.adj_mat = torch.tensor(self.adata.obsp['spatial_connectivities'].todense())

    def build_neighborhood_from_hops(self, idx):
        # Get nn indexes for the n_hop required
        nn_index_list = self.neighbors_dict[idx]
        # Get the gene expression values for the nn
        exp_matrix = self.expression_mtx[nn_index_list].type('torch.FloatTensor') # Original dtype was 'torch.float64'
        # Get image embeddings if needed
        embeddings = self.embeddings[nn_index_list] if self.use_visual_features else None

        if not self.pre_masked:
            # Get median imputation mask for nn
            median_imp_mask = self.median_imp_mask[nn_index_list] #size 7 x 128
            neigh_data = (exp_matrix, median_imp_mask, embeddings)

        else:
            pre_masked_exp = self.pre_masked_expression_mtx[nn_index_list].type('torch.FloatTensor')
            random_mask = self.random_mask[nn_index_list] #size 7 x 128
            neigh_data = (exp_matrix, pre_masked_exp, random_mask, embeddings)

        return neigh_data

    def build_neighborhood_from_distance(self, idx):
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
        # Get expression and median imputation mask of neighborhoods
        if not self.pre_masked:
            if self.neighborhood_type == 'circular_hops':
                item['exp_matrix_gt'], item['real_missing'], item['visual_features'] = self.build_neighborhood_from_hops(idx)
            else: # When self.neighborhood_type == 'nn_distance'
                item['exp_matrix_gt'], item['real_missing'], item['visual_features'] = self.build_neighborhood_from_distance(idx)

        else:
            if self.neighborhood_type == 'circular_hops':
                item['exp_matrix_gt'], item['pre_masked_exp'], item['random_mask'], item['visual_features'] = self.build_neighborhood_from_hops(idx)
            else: # When self.neighborhood_type == 'nn_distance'
                item['exp_matrix_gt'], item['pre_masked_exp'], item['random_mask'], item['visual_features'] = self.build_neighborhood_from_distance(idx)
                
        if not self.use_visual_features:
            del item['visual_features']

        return item

    def __len__(self):
        return len(self.adata)