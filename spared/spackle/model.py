import torch
from torch import nn
import torch.nn.functional as F
from spackle.utils import *
import lightning as L
import pathlib
import sys

# El path a spared es ahora diferente
SPARED_PATH = pathlib.Path(__file__).resolve().parent.parent

# Agregar el directorio padre al sys.path para los imports
sys.path.append(str(SPARED_PATH))
# Import im_encoder.py file
from metrics.metrics import get_metrics
# Remove the path from sys.path
sys.path.remove(str(SPARED_PATH))


class MLP(torch.nn.Module):
    def __init__(self, num_layers, mlp_dim, output_dim, act, redim_layer = False):
        """
        Generic multilayer perceptron where the dimensions of layers can be specified along with the activation function.

        Args:
            num_layers (int):       Number of the layers stacked in the MLP. 
            mlp_dim (int):          Expected dimension of the input data for the MLP.
            output_dim (int)        Dimension that the output data should have. Same as number of genes.
            act (str):              String with the activation function to use. Options are all activation functions defined in: 
                                    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
                                    examples are: 'ELU', 'ReLU', 'Hardtanh'... the string is case sensitive with respect to the ones
                                    defined at PyTorch website.
            redim_layer (bool):     Whether or not to add a linear layer before the MLP to redimension input data to mlp_dim.
        """
        super(MLP, self).__init__()

        # Activation function definition
        self.activation = getattr(nn, act)()

        # Define hidden sizes list
        self.num_layers = num_layers
        # Define the dimension of the input data
        self.mlp_dim = mlp_dim
        # Define the dimension of the output data
        self.output_dim = output_dim

        # Define layers
        self.redim_layer = redim_layer
        if self.redim_layer:
            self.pre_mlp_linear = nn.Linear(self.output_dim, self.mlp_dim)
        # The output layer is the last element in self.layers
        self.layers = nn.ModuleList()
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                self.layers.append(nn.Linear(self.mlp_dim, self.mlp_dim))

        self.layers.append(nn.Linear(self.mlp_dim, self.output_dim))
    
    def forward(self, expression):
        """
        Performs forward pass of the MLP.

        Args:
            expression (tensor): Matrix of shape (batch_size, num_neighbors, features)

        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, output_size). Output_size is the last element in self.mlp_dims  
        """

        x = expression

        # Redim input data if necessary
        if self.redim_layer:
            x = self.pre_mlp_linear(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        out = x

        return out
    
class Autoencoder(torch.nn.Module):
    def __init__(self, ae_layer_dims, output_dim, act):
        """
        Autoencoder-like architecture where the number of layers and their dimensions are specified by a list of dims.

        Args:
            ae_layer_dims (list):   Dimensions of the layers for the autoencoder.
            output_dim (int)        Dimension that the output data should have. Same as number of genes.
            act (str):              String with the activation function to use. Options are all activation functions defined in: 
                                    https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
                                    examples are: 'ELU', 'ReLU', 'Hardtanh'... the string is case sensitive with respect to the ones
                                    defined at PyTorch website.
        """
        super(Autoencoder, self).__init__()

        # Activation function definition
        self.activation = getattr(nn, act)()

        # Define the dimension of the autoencoder layers
        self.ae_layer_dims = ae_layer_dims
        # Define the dimension of the output data
        self.output_dim = output_dim

        # Define layer that redims input tensor to the ae's input dimension
        self.pre_mlp_linear = nn.Linear(self.output_dim, self.ae_layer_dims[0])
        
        # Define autoencoder
        self.layers = nn.ModuleList()
        for i in range(len(self.ae_layer_dims)-1):
            self.layers.append(nn.Linear(self.ae_layer_dims[i], self.ae_layer_dims[i+1]))

        # Redim output to gene numbers
        self.output_layer = nn.Linear(self.ae_layer_dims[-1], self.output_dim)
    
    def forward(self, expression):
        """
        Performs forward pass of the autoencoder.

        Args:
            expression (tensor): Matrix of shape (batch_size, num_neighbors, features)

        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, output_size). Output_size is the last element in self.mlp_dims  
        """

        x = expression
        
        # Redim input data 
        x = self.pre_mlp_linear(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.output_layer(x)

        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, transformer_dim, transformer_heads, transformer_encoder_layers, data_input_size, use_out_linear_layer=True):
        """
        Generic transformer architecture (encoder + decoder) from pytorch with optional linear layers before and after the transformer.

        Args:
            layer_list (list):  
            act (str):          
        """
        super(TransformerEncoder, self).__init__()

        # Define architecture of transformer
        self.d_model = transformer_dim
        self.nhead = transformer_heads
        self.num_enc_layers = transformer_encoder_layers
        self.input_size = data_input_size
        self.use_out_linear_layer = use_out_linear_layer
        
        # Declare the structure of a single encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        # Declare the stack of N encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_enc_layers
            )
        
        self.linear = nn.Linear(self.input_size, self.d_model)
        self.out_linear = nn.Linear(self.d_model, self.input_size) if self.use_out_linear_layer else None
    
    def forward(self, x):
        """
        Performs forward pass of the TransformerEncoder.

        Args:
            x (tensor): Matrix of shape (batch_size, n_spots, genes) with the gene expression of neighboring patches and a mask over some randomly selected values.
            
        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, n_spots, genes). Gene expression matrix with missing values predicted.
        """
        x = self.linear(x)
        x = self.transformer_encoder(x)
        if self.use_out_linear_layer:
            x = self.out_linear(x)
        return x

class TransformerEncoderVisualFeatures(torch.nn.Module):
    def __init__(self, transformer_dim, transformer_heads, transformer_encoder_layers, n_genes, vis_features_dim, include_genes):
        """
        Generic transformer encoder architecture from pytorch with optional linear layers before and after the transformer, and an
        adapter for considering visual features
         
        """
        super(TransformerEncoderVisualFeatures, self).__init__()

        # Define architecture of transformer
        self.d_model = transformer_dim
        self.nhead = transformer_heads
        self.num_enc_layers = transformer_encoder_layers
        self.n_genes = n_genes
        self.vis_features_dim = vis_features_dim
        self.include_genes = include_genes

        # When include_genes is True, the input size is the sum of the number of genes and the visual features dimension 
        # As the visual features are reduced to the number of genes, the input size is 2 times n_genes
        if self.include_genes:
            self.input_size = 2 * self.n_genes
        else: 
            self.input_size = self.n_genes
        self.output_size = self.n_genes
        
        # Declare linear layer to reduce visual features dimension
        self.linear_vis = nn.Linear(self.vis_features_dim, self.n_genes)
        # Declare linear layer to reduce input size to d_model
        self.in_linear = nn.Linear(self.input_size, self.d_model)
        # Declare the structure of a single encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        # Declare the stack of N encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=self.num_enc_layers
            )
        # Declare linear layer to increase output size to n_genes
        self.out_linear = nn.Linear(self.d_model, self.output_size)
    
    def forward(self, input_data):
        """
        Performs forward pass of the TransformerEncoder.

        Args:
            input_genes (tensor): Matrix of shape (batch_size, n_spots, genes) with the gene expression of neighboring patches and a mask over some randomly selected values.
            visual_features (tensor): Matrix of shape (batch_size, n_spots, visual_features_dim) with the visual features of neighboring patches.
            
        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, n_spots, genes). Gene expression matrix with missing values predicted.
        """
        # NOTE: if self.args.include_genes==False the input genes parameter is ignored by the model
        input_genes, visual_features = input_data
        visual_features = self.linear_vis(visual_features)
        
        if self.include_genes:
            x = torch.cat((input_genes, visual_features), dim=-1)
        else:
            x = visual_features
        
        x = self.in_linear(x)
        x = self.transformer_encoder(x)
        x = self.out_linear(x)
        
        return x

class DoubleTransformerEncoder(torch.nn.Module):
    def __init__(self, transformer_dim, transformer_heads, transformer_encoder_layers, n_genes, vis_features_dim, num_mlp_layers, mlp_act_function, matrix_union_method = 'concatenate'):
        """
        Generic transformer encoder architecture from pytorch with optional linear layers before and after the transformer, and an
        adapter for considering visual features
         
        """
        super(DoubleTransformerEncoder, self).__init__()
        self.matrix_union_method = matrix_union_method # Either 'concatenate' or 'sum'

        # Define architecture of transformer
        self.d_model = transformer_dim
        self.nhead = transformer_heads
        self.num_enc_layers = transformer_encoder_layers
        self.n_genes = n_genes
        self.vis_features_dim = vis_features_dim
        
        ## Define architecture of the gene processing branch
        self.genes_transformer = TransformerEncoder(
                self.d_model,
                self.nhead,
                self.num_enc_layers,
                self.n_genes,
                use_out_linear_layer=False
            )

        ## Define architecture of the visual features processing branch
        self.visual_fts_transformer = TransformerEncoder(
                self.d_model,
                self.nhead,
                self.num_enc_layers,
                self.vis_features_dim,
                use_out_linear_layer=False
            )
        
        ## Define architecture of the MLP
        self.num_mlp_layers = num_mlp_layers
        self.mlp_act = mlp_act_function
        if self.matrix_union_method == 'concatenate':
            self.mlp_input_dim = self.d_model * 2
        else:
            self.mlp_input_dim = self.d_model

        self.mlp = MLP(num_layers=self.num_mlp_layers, 
                       mlp_dim=self.mlp_input_dim, #dim of concatenated transformer outputs
                       output_dim=self.n_genes, 
                       act=self.mlp_act)
    
    def forward(self, input_data):
        """
        Performs forward pass of the TransformerEncoder.

        Args:
            input_genes (tensor): Matrix of shape (batch_size, n_spots, genes) with the gene expression of neighboring patches and a mask over some randomly selected values.
            visual_features (tensor): Matrix of shape (batch_size, n_spots, visual_features_dim) with the visual features of neighboring patches.
            
        Returns:
            tensor: A tensor matrix where dimensions are (batch_size, n_spots, genes). Gene expression matrix with missing values predicted.
        """
        # NOTE: if self.args.include_genes==False the input genes parameter is ignored by the model
        input_genes, visual_features = input_data

        # Process input_genes with the gene processing branch
        input_genes = self.genes_transformer(input_genes)

        # Process visual_features with the visual features processing branch
        visual_features = self.visual_fts_transformer(visual_features)

        # Combine the output matrices of both branches following the method established in the argparse
        if self.matrix_union_method == 'concatenate':
            x = torch.cat((input_genes, visual_features), dim=-1)
        else:
            x = input_genes + visual_features

        # Pass the final matrix through the MLP
        x = self.mlp(x)
        
        return x

class GeneImputationModel(L.LightningModule):
    def __init__(
        self,
        args,
        data_input_size, # 128 (or number of genes) if input shape is [batch, 7 (n_spots), 128]
        lr,
        optimizer,
        train_mask_prob_tensor = None, # None in case the model is used for inference and doesn't require random masking probabilities
        val_test_mask_prob_tensor = None,
        vis_features_dim = 0
        ):
        super().__init__()
        self.args = args
        self.train_mask_prob_tensor = train_mask_prob_tensor
        self.val_test_mask_prob_tensor = val_test_mask_prob_tensor
        self.base_architecture = args.base_arch
        self.use_visual_features = args.use_visual_features
        self.use_double_branch_archit = args.use_double_branch_archit
        self.lr = lr
        self.optimizer = optimizer
        self.transformer_dim = data_input_size
        # Set split name for testing
        self.split_name = None

        # Define loss criterion
        self.criterion = torch.nn.MSELoss()

        # Define outputs of the validation, test and train step
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Auxiliary variables to log best metrics
        self.best_metrics = None
        min_max_metric_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}
        self.metric_objective = min_max_metric_dict[self.args.optim_metric]
    
        # Define architecture of the model
        if self.use_visual_features:
            if self.use_double_branch_archit:
                self.model = DoubleTransformerEncoder(
                    transformer_dim = self.transformer_dim, 
                    transformer_heads = args.transformer_heads, 
                    transformer_encoder_layers = args.transformer_encoder_layers, 
                    n_genes = data_input_size,
                    vis_features_dim = vis_features_dim, 
                    num_mlp_layers = args.num_mlp_layers, 
                    mlp_act_function = args.mlp_act, 
                    matrix_union_method = args.matrix_union_method)
            
            else:
                self.model = TransformerEncoderVisualFeatures(
                    self.transformer_dim,
                    args.transformer_heads,
                    args.transformer_encoder_layers,
                    n_genes = data_input_size,
                    vis_features_dim = vis_features_dim, 
                    include_genes = self.args.include_genes
                )
          
        else:
            if self.base_architecture == 'transformer_encoder':
                self.model = TransformerEncoder(
                    self.transformer_dim,
                    args.transformer_heads,
                    args.transformer_encoder_layers,
                    data_input_size
                )

            else: # base_arch == 'MLP'
                self.model = Autoencoder(
                    ae_layer_dims = args.ae_layer_dims,
                    output_dim=data_input_size, 
                    act=args.mlp_act)

    def forward(self, input_data):
        model_output = self.model(input_data)
        return model_output

    def pred_outputs_from_batch(self, batch, pre_masked_batch = False):
        del batch['split_name']
        # Extract batch variables
        batch = {k: v.to(self.device) for k, v in batch.items()}
        expression_gt = batch['exp_matrix_gt']
        
        ## Randomly mask the gene expression mask if it hasn't been masked yet
        if not pre_masked_batch:
            # Input expression matrix
            input_genes = batch['exp_matrix_gt'].clone()
            # Create random mask based on the masking probability of each gene
            random_mask = torch.rand(input_genes.shape).to(self.device) < self.train_mask_prob_tensor
            # Combine random mask with the median imputation mask
            random_mask = random_mask.to(self.device) & batch['real_missing']
            # Mask input expression matrix
            input_genes[random_mask] = 0

        else:
            # Get pre-masked genes
            input_genes = batch['pre_masked_exp']
            # Get random_mask used to previously mask the input data
            random_mask = batch['random_mask']
        
        # Get visual features and include the matrix in input_data if args.use_visual_features == True
        if self.use_visual_features:
            vis_features = batch['visual_features']
            input_data = (input_genes, vis_features)
        
        else:
            input_data = input_genes
        
        # Get predictions
        prediction = self.forward(input_data)

        return prediction, expression_gt, random_mask

    def training_step(self, batch):
        # Get the outputs from model
        prediction, expression_gt, random_mask = self.pred_outputs_from_batch(batch=batch)
        # Compute expression MSE loss
        loss = self.criterion(prediction, expression_gt)
        # Log loss per training step
        self.log_dict({'train_loss': loss}, on_step=True)

        # Get all variables only for first node and append train step outputs 
        self.training_step_outputs.append((prediction[:,0,:], expression_gt[:,0,:], random_mask[:,0,:]))

        return loss

    def on_train_epoch_end(self):
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.training_step_outputs)
        # Convert outputs into tensors and concatenate the results of each step
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Calculate and log metrics
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put train prefix in metric dict
        metrics = {f'train_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)
        
        # Free memory
        self.training_step_outputs.clear()
        
    def validation_step(self, batch):
        # Get the outputs from model
        prediction, expression_gt, random_mask = self.pred_outputs_from_batch(batch=batch)

        # Get all variables only for first node and append val step outputs 
        self.validation_step_outputs.append((prediction[:,0,:], expression_gt[:,0,:], random_mask[:,0,:]))

    def on_validation_epoch_end(self):
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.validation_step_outputs)
        # Convert outputs into tensors and concatenate the results of each step
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put val prefix in metric dict
        metrics = {f'val_{k}': v for k, v in metrics.items()}

        # Auxiliar metric dict with a changed name to facilitate things. aux_metrics is not necesarily representing best metrics.
        aux_metrics = {f'best_{k}': v for k, v in metrics.items()}
        # Log best metrics
        if self.best_metrics is None:
            self.best_metrics = aux_metrics
        else:
            # Define metric name
            metric_name = f'best_val_{self.args.optim_metric}'
            # Determine if we got a new best model (robust to minimization or maximization of any metric)
            got_best_min = (self.metric_objective == 'min') and (aux_metrics[metric_name] < self.best_metrics[metric_name])
            got_best_max = (self.metric_objective == 'max') and (aux_metrics[metric_name] > self.best_metrics[metric_name])
            # If we got a new best model, save it and log the metrics in wandb
            if got_best_min or got_best_max:
                self.best_metrics = aux_metrics
        
        # Log metrics and best metrics in each validation step
        self.log_dict({**metrics, **self.best_metrics})
        
        # Free memory
        self.validation_step_outputs.clear()  

    # Run this test_step when model should expect testing data to be pre-masked (when model is compared to the median method)
    def test_step(self, batch):
        # Save testing split name for logging purposes
        self.split_name = batch['split_name'][0]
        # Get the outputs from model
        prediction, expression_gt, random_mask = self.pred_outputs_from_batch(batch=batch, pre_masked_batch=True)

        # Get all variables only for first node and append test step outputs 
        self.test_step_outputs.append((prediction[:,0,:], expression_gt[:,0,:], random_mask[:,0,:]))

    # Run this test_step when model should NOT expect testing data to be pre-masked (when model masking is done during the test_step)
    '''def test_step(self, batch):
        # Save testing split name for logging purposes
        self.split_name = batch['split_name'][0]
        # Get the outputs from model
        prediction, expression_gt, random_mask = self.pred_outputs_from_batch(batch=batch)

        # Get all variables only for first node and append test step outputs 
        self.test_step_outputs.append((prediction[:,0,:], expression_gt[:,0,:], random_mask[:,0,:]))'''

    def on_test_epoch_end(self):
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.test_step_outputs)
        # Convert outputs into tensors and concatenate the results of each step
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put test prefix in metric dict
        metrics = {f'testing_{self.split_name}_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)
        
        # Free memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        try:
            optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.lr, momentum=self.args.momentum)
        except:
            optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.lr)

        return optimizer
    