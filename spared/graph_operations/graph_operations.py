import anndata as ad
import torch
from positional_encodings.torch_encodings import PositionalEncoding2D
from tqdm import tqdm
from torch_geometric.data import Data as geo_Data
import numpy as np
import pathlib
import squidpy as sq
from torch_geometric.utils import from_scipy_sparse_matrix
from typing import Tuple
import sys
from typing import Tuple

# Path a spared 
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from filtering import filtering
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))

### Graph building functions:
def get_graphs_one_slide(adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool) -> Tuple[dict,int]:
    """ Get neighbor graphs for a single slide.
    This function receives an AnnData object with a single slide and for each node computes the graph in an
    n_hops radius in a pytorch geometric format. The AnnData object must have both embeddings and predictions in the
    adata.obsm attribute.

    It returns a dictionary where the patch names are the keys and a pytorch geometric graph for each one as
    values. NOTE: The first node of every graph is the center.

    Args:
        adata (ad.AnnData): The AnnData object with the slide data.
        n_hops (int): The number of hops to compute the graph.
        layer (str): The layer of the graph to predict. Will be added as y to the graph.
        hex_geometry (bool): Whether the slide has hexagonal geometry or not.

    Returns:
        Tuple(dict,int)
        dict: A dictionary where the patch names are the keys and pytorch geometric graph for each one as values. The first node of every graph is the center.
        int: Max column or row difference between the center and the neighbors. Used for positional encoding.                   
    """
    # Compute spatial_neighbors
    if hex_geometry:
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6) # Hexagonal visium case
    else:
        sq.gr.spatial_neighbors(adata, coord_type='grid', n_neighs=8) # Grid STNet dataset case

    # Get the adjacency matrix
    adj_matrix = adata.obsp['spatial_connectivities']

    # Define power matrix
    power_matrix = adj_matrix.copy()
    # Define the output matrix
    output_matrix = adj_matrix.copy()

    # Iterate through the hops
    for i in range(n_hops-1):
        # Compute the next hop
        power_matrix = power_matrix * adj_matrix
        # Add the next hop to the output matrix
        output_matrix = output_matrix + power_matrix

    # Zero out the diagonal
    output_matrix.setdiag(0)
    # Threshold the matrix to 0 and 1
    output_matrix = output_matrix.astype(bool).astype(int)

    # Define dict from index to obs name
    index_to_obs = {i: obs for i, obs in enumerate(adata.obs.index.values)}

    # Define neighbors dicts (one with names and one with indexes)
    neighbors_dict_index = {}
    neighbors_dict_names = {}
    matrices_dict = {}

    # Iterate through the rows of the output matrix
    for i in range(output_matrix.shape[0]):
        # Get the non-zero elements of the row
        non_zero_elements = output_matrix[i].nonzero()[1]
        # Get the names of the neighbors
        non_zero_names = [index_to_obs[index] for index in non_zero_elements]
        # Add the neighbors to the neighbors dicts. NOTE: the first index is the query obs
        neighbors_dict_index[i] = [i] + list(non_zero_elements)
        neighbors_dict_names[index_to_obs[i]] = np.array([index_to_obs[i]] + non_zero_names)
        
        # Subset the matrix to the non-zero elements and store it in the matrices dict
        matrices_dict[index_to_obs[i]] = output_matrix[neighbors_dict_index[i], :][:, neighbors_dict_index[i]]

    
    ### Get pytorch geometric graphs ###
    layers_dict = {key: torch.from_numpy(adata.layers[key]).type(torch.float32) for key in adata.layers.keys()} # Get global layers
    pos = torch.from_numpy(adata.obs[['array_row', 'array_col']].values)                                        # Get global positions

    # Get embeddings and predictions keys
    emb_key_list = [k for k in adata.obsm.keys() if 'embeddings' in k]
    pred_key_list = [k for k in adata.obsm.keys() if 'predictions' in k]
    assert len(emb_key_list) == 1, 'There are more than 1 or no embedding keys in adata.obsm'
    assert len(pred_key_list) == 1, 'There are more than 1 or no prediction keys in adata.obsm'
    emb_key, pred_key = emb_key_list[0], pred_key_list[0]

    # If embeddings and predictions are present in obsm, get them
    embeddings = torch.from_numpy(adata.obsm[emb_key]).type(torch.float32)
    predictions = torch.from_numpy(adata.obsm[pred_key]).type(torch.float32)

    # If layer contains delta then add a used_mean attribute to the graph
    used_mean = torch.from_numpy(adata.var[f'{layer}_avg_exp'.replace('deltas', 'log1p')].values).type(torch.float32) if 'deltas' in layer else None

    # Define the empty graph dict
    graph_dict = {}
    max_abs_d_pos=-1

    # Cycle over each obs
    for i in tqdm(range(len(neighbors_dict_index)), leave=False, position=1):
        central_node_name = index_to_obs[i]                                                 # Get the name of the central node
        curr_nodes_idx = torch.tensor(neighbors_dict_index[i])                              # Get the indexes of the nodes in the graph
        curr_adj_matrix = matrices_dict[central_node_name]                                  # Get the adjacency matrix of the graph (precomputed)
        curr_edge_index, _ = from_scipy_sparse_matrix(curr_adj_matrix)                      # Get the edge index and edge attribute of the graph
        curr_layers = {key: layers_dict[key][curr_nodes_idx] for key in layers_dict.keys()} # Get the layers of the graph filtered by the nodes
        curr_pos = pos[curr_nodes_idx]                                                      # Get the positions of the nodes in the graph
        curr_d_pos = curr_pos - curr_pos[0]                                                 # Get the relative positions of the nodes in the graph

        # Define the graph
        graph_dict[central_node_name] = geo_Data(
            y=curr_layers[layer],
            edge_index=curr_edge_index,
            pos=curr_pos,
            d_pos=curr_d_pos,
            embeddings=embeddings[curr_nodes_idx],
            predictions=predictions[curr_nodes_idx] if predictions is not None else None,
            used_mean=used_mean if used_mean is not None else None,
            num_nodes=len(curr_nodes_idx),
            mask=layers_dict['mask'][curr_nodes_idx]
        )

        max_curr_d_pos=curr_d_pos.abs().max()
        if max_curr_d_pos>max_abs_d_pos:
            max_abs_d_pos=max_curr_d_pos

    #cast as int
    max_abs_d_pos=int(max_abs_d_pos)
    
    # Return the graph dict
    return graph_dict, max_abs_d_pos

def get_sin_cos_positional_embeddings(graph_dict: dict, max_d_pos: int) -> dict:
    """ Get positional encodings for a neighbor graph.
    This function adds a transformer-like positional encodings to each graph in a graph dict. It adds the positional
    encodings under the attribute 'positional_embeddings' for each graph. 

    Args:
        graph_dict (dict): A dictionary where the patch names are the keys and a pytorch geometric graphs for each one are values.
        max_d_pos (int): Max absolute value in the relative position matrix.

    Returns:
        dict: The input graph dict with the information of positional encodings for each graph.
    """
    graph_dict_keys = list(graph_dict.keys())
    embedding_dim = graph_dict[graph_dict_keys[0]].embeddings.shape[1]

    # Define the positional encoding model
    p_encoding_model= PositionalEncoding2D(embedding_dim)

    # Define the empty grid with size (batch_size, x, y, channels)
    grid_size = torch.zeros([1, 2*max_d_pos+1, 2*max_d_pos+1, embedding_dim])

    # Obtain the embeddings for each position
    positional_look_up_table = p_encoding_model(grid_size)        

    for key, value in graph_dict.items():
        d_pos = value.d_pos
        grid_pos = d_pos + max_d_pos
        graph_dict[key].positional_embeddings = positional_look_up_table[0,grid_pos[:,0],grid_pos[:,1],:]
    
    return graph_dict

def get_graphs(adata: ad.AnnData, n_hops: int, layer: str, hex_geometry: bool=True) -> dict:
    """ Get graphs for all the slides in a dataset.
    This function wraps the get_graphs_one_slide function to get the graphs for all the slides in the dataset.
    After computing the graph dicts for each slide it concatenates them into a single dictionary which is then used to compute
    the positional embeddings for each graph.

    For details see get_graphs_one_slide and get_sin_cos_positional_embeddings functions.

    Args:
        adata (ad.AnnData): The AnnData object used to build the graphs.
        n_hops (int): The number of hops to compute each graph.
        layer (str): The layer of the graph to predict. Will be added as y to the graph.
        hex_geometry (bool): Whether the graph is hexagonal or not. Only true for visium datasets. Defaults to True.

    Returns:
        dict: A dictionary where the spots' names are the keys and pytorch geometric graphs are values.
    """

    print('Computing graphs...')

    # Get unique slide ids
    unique_ids = adata.obs['slide_id'].unique()

    # Global dictionary to store the graphs (pytorch geometric graphs)
    graph_dict = {}
    max_global_d_pos=-1

    # Iterate through slides
    for slide in tqdm(unique_ids, leave=True, position=0):
        curr_adata = filtering.get_slide_from_collection(adata, slide)
        curr_graph_dict, max_curr_d_pos = get_graphs_one_slide(curr_adata, n_hops, layer, hex_geometry)
        
        # Join the current dictionary to the global dictionary
        graph_dict = {**graph_dict, **curr_graph_dict}

        if max_curr_d_pos>max_global_d_pos:
            max_global_d_pos=max_curr_d_pos
    
    graph_dict = get_sin_cos_positional_embeddings(graph_dict, max_global_d_pos)

    # Return the graph dict
    return graph_dict
