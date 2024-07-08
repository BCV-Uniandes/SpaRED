import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import lightning as L
from torchvision.transforms import Compose, RandomApply, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, Normalize
from metrics import get_metrics


class ImageEncoder(torch.nn.Module):
    def __init__(self, backbone, use_pretrained,  latent_dim):

        super(ImageEncoder, self).__init__()

        self.backbone = backbone
        self.use_pretrained = use_pretrained
        self.latent_dim = latent_dim

        # Initialize the model using various options 
        self.encoder, self.input_size = self.initialize_model()

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        model_weights = 'IMAGENET1K_V1' if self.use_pretrained else None
        input_size = 0

        if self.backbone == "resnet": ##
            """ Resnet18 acc@1 (on ImageNet-1K): 69.758
            """
            model_ft = models.resnet18(weights=model_weights)   #Get model
            num_ftrs = model_ft.fc.in_features                  #Get in features of the fc layer (final layer)
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)  #Keep in features, but modify out features for self.latent_dim
            input_size = 224                                    #Set input size of each image

        elif self.backbone == "resnet50":
            """ Resnet50 acc@1 (on ImageNet-1K): 76.13
            """
            model_ft = models.resnet50(weights=model_weights)   
            num_ftrs = model_ft.fc.in_features                  
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)  
            input_size = 224                                    

        elif self.backbone == "ConvNeXt":
            """ ConvNeXt tiny acc@1 (on ImageNet-1K): 82.52
            """
            model_ft = models.convnext_tiny(weights=model_weights)
            num_ftrs = model_ft.classifier[2].in_features
            model_ft.classifier[2] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "EfficientNetV2":
            """ EfficientNetV2 small acc@1 (on ImageNet-1K): 84.228
            """
            model_ft = models.efficientnet_v2_s(weights=model_weights)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 384

        elif self.backbone == "InceptionV3":
            """ InceptionV3 acc@1 (on ImageNet-1K): 77.294
            """
            model_ft = models.inception_v3(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 299

        elif self.backbone == "MaxVit":
            """ MaxVit acc@1 (on ImageNet-1K): 83.7
            """
            model_ft = models.maxvit_t(weights=model_weights)
            num_ftrs = model_ft.classifier[5].in_features
            model_ft.classifier[5] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "MobileNetV3":
            """ MobileNet V3 acc@1 (on ImageNet-1K): 67.668
            """
            model_ft = models.mobilenet_v3_small(weights=model_weights)
            num_ftrs = model_ft.classifier[3].in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ResNetXt":
            """ ResNeXt-50 32x4d acc@1 (on ImageNet-1K): 77.618
            """
            model_ft = models.resnext50_32x4d(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224


        elif self.backbone == "ShuffleNetV2":
            """ ShuffleNetV2 acc@1 (on ImageNet-1K): 60.552
            """
            model_ft = models.shufflenet_v2_x0_5(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ViT":
            """ Vision Transformer acc@1 (on ImageNet-1K): 81.072
            """
            model_ft = models.vit_b_16(weights=model_weights)
            num_ftrs = model_ft.heads.head.in_features
            model_ft.heads.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "WideResNet":
            """ Wide ResNet acc@1 (on ImageNet-1K): 78.468
            """
            model_ft = models.wide_resnet50_2(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "densenet": 
            """ Densenet acc@1 (on ImageNet-1K): 74.434
            """
            model_ft = models.densenet121(weights=model_weights)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224
        
        elif self.backbone == "swin": 
            """ Swin Transformer tiny acc@1 (on ImageNet-1K): 81.474
            """
            model_ft = models.swin_t(weights=model_weights)
            num_ftrs = model_ft.head.in_features
            model_ft.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def forward(self, tissue_tiles):

        latent_space = self.encoder(tissue_tiles)

        return latent_space
    


class ImageBackbone(L.LightningModule):
    def __init__(self, args,  latent_dim):

        super(ImageBackbone, self).__init__()
        
        # Define normal hyperparameters
        self.save_hyperparameters()
        self.args = args
        self.backbone = args.img_backbone
        self.use_pretrained = args.img_use_pretrained
        self.latent_dim = latent_dim

        # Define image transformations        
        self.train_transforms = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   
                                        RandomHorizontalFlip(p=0.5),
                                        RandomVerticalFlip(p=0.5),
                                        RandomApply([RandomRotation((90, 90))], p=0.5)])
        if args.average_test:
            self.test_transforms = Compose([Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), EightSymmetry()])
        else:
            self.test_transforms = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Define loss criterion
        self.criterion = torch.nn.MSELoss()

        # Initialize the model using various options 
        self.encoder, self.input_size = self.initialize_model()

        # Define outputs of the validation, test and train step
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        # Auxiliary variables to log best metrics
        self.best_metrics = None
        min_max_metric_dict = {'PCC-Gene': 'max', 'PCC-Patch': 'max', 'MSE': 'min', 'MAE': 'min', 'R2-Gene': 'max', 'R2-Patch': 'max', 'Global': 'max'}
        self.metric_objective = min_max_metric_dict[self.args.optim_metric]

    def initialize_model(self):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        model_weights = 'IMAGENET1K_V1' if self.use_pretrained else None
        input_size = 0

        if self.backbone == "resnet": ##
            """ Resnet18 acc@1 (on ImageNet-1K): 69.758
            """
            model_ft = models.resnet18(weights=model_weights)   #Get model
            num_ftrs = model_ft.fc.in_features                  #Get in features of the fc layer (final layer)
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)  #Keep in features, but modify out features for self.latent_dim
            input_size = 224                                    #Set input size of each image

        elif self.backbone == "ConvNeXt":
            """ ConvNeXt tiny acc@1 (on ImageNet-1K): 82.52
            """
            model_ft = models.convnext_tiny(weights=model_weights)
            num_ftrs = model_ft.classifier[2].in_features
            model_ft.classifier[2] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "EfficientNetV2":
            """ EfficientNetV2 small acc@1 (on ImageNet-1K): 84.228
            """
            model_ft = models.efficientnet_v2_s(weights=model_weights)
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 384

        elif self.backbone == "InceptionV3":
            """ InceptionV3 acc@1 (on ImageNet-1K): 77.294
            """
            model_ft = models.inception_v3(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 299

        elif self.backbone == "MaxVit":
            """ MaxVit acc@1 (on ImageNet-1K): 83.7
            """
            model_ft = models.maxvit_t(weights=model_weights)
            num_ftrs = model_ft.classifier[5].in_features
            model_ft.classifier[5] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "MobileNetV3":
            """ MobileNet V3 acc@1 (on ImageNet-1K): 67.668
            """
            model_ft = models.mobilenet_v3_small(weights=model_weights)
            num_ftrs = model_ft.classifier[3].in_features
            model_ft.classifier[3] = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ResNetXt":
            """ ResNeXt-50 32x4d acc@1 (on ImageNet-1K): 77.618
            """
            model_ft = models.resnext50_32x4d(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ShuffleNetV2":
            """ ShuffleNetV2 acc@1 (on ImageNet-1K): 60.552
            """
            model_ft = models.shufflenet_v2_x0_5(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "ViT":
            """ Vision Transformer acc@1 (on ImageNet-1K): 81.072
            """
            model_ft = models.vit_b_16(weights=model_weights)
            num_ftrs = model_ft.heads.head.in_features
            model_ft.heads.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "WideResNet":
            """ Wide ResNet acc@1 (on ImageNet-1K): 78.468
            """
            model_ft = models.wide_resnet50_2(weights=model_weights)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        elif self.backbone == "densenet": 
            """ Densenet acc@1 (on ImageNet-1K): 74.434
            """
            model_ft = models.densenet121(weights=model_weights)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224
        
        elif self.backbone == "swin": 
            """ Swin Transformer tiny acc@1 (on ImageNet-1K): 81.474
            """
            model_ft = models.swin_t(weights=model_weights)
            num_ftrs = model_ft.head.in_features
            model_ft.head = nn.Linear(num_ftrs, self.latent_dim)
            input_size = 224

        # elif self.backbone == "densenet121-kather100k":
        #     """ Densenet F1 (Kather-100k): 0.993
        #     """
        #     histo_model = PatchPredictor(pretrained_model="densenet121-kather100k", batch_size=self.args.batch_size).model
        #     num_ftrs = histo_model.classifier.in_features
        #     histo_model.classifier = nn.Linear(num_ftrs, self.latent_dim)
        #     histo_state_dict = histo_model.state_dict()
            
        #     # Replace some keys that are modified by the TIA toolbox
        #     histo_state_dict = {k.replace('feat_extract', 'features'): v for k, v in histo_state_dict.items()}
            
        #     model_ft = models.densenet121()
        #     model_ft.classifier = nn.Linear(num_ftrs, self.latent_dim)
        #     model_ft.load_state_dict(histo_state_dict)
            
        #     # NOTE This is for linear probing
        #     # # Freeze all layers except the last one
        #     # for param in model_ft.parameters():
        #     #     param.requires_grad = False
            
        #     # for param in model_ft.classifier.parameters():
        #     #     param.requires_grad = True

        #     input_size = 224
        # else:
        #     print("Invalid model name, exiting...")
        #     exit()

        return model_ft, input_size

    def forward(self, tissue_tiles):

        latent_space = self.encoder(tissue_tiles)

        return latent_space
    
    def find_batch_patch_key(self, batch):
        # Find the key of dataset.obsm that contains the patches
        patch_key = [k for k in batch.obsm.keys() if 'patches' in k]
        # Assert that there is only one key
        assert len(patch_key) == 1, 'There should be only one key with patches in data.obsm'
        patch_key = patch_key[0]
        return patch_key

    def pred_outputs_from_batch(self, batch):

        # Get the patch key in the batch
        patch_key = self.find_batch_patch_key(batch)
        
        # Get (and reshape) images from batch
        tissue_tiles = batch.obsm[patch_key]
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        
        # Permute dimensions to be in correct order for normalization
        tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
        # Make transformations in tissue tiles
        tissue_tiles = tissue_tiles/255.
        # Transform tiles
        tissue_tiles = self.test_transforms(tissue_tiles)
        
        # Get groundtruth of expression
        expression_gt = batch.X

        # Get output of the model
        # If tissue tiles is tuple then we will compute outputs of the 8 symmetries and then average them for prediction
        if isinstance(tissue_tiles, tuple):
            pred_list = [self.forward(tissue_rot) for tissue_rot in tissue_tiles]
            pred_stack = torch.stack(pred_list)
            expression_pred = pred_stack.mean(dim=0)
        # If tissue tiles is not tuple then a single prediction is done with the original image
        else:
            expression_pred = self.forward(tissue_tiles)
        
        # Handle delta vs absolute prediction with means
        # If the adata object has a used mean attribute then we will use it to unnormalize the data
        general_adata = batch.adatas[0]
        if 'used_mean' in general_adata.var.keys():
            means = general_adata.var['used_mean'].values
            # Pass means to torch tensor in the same device as the model
            means = torch.tensor(means, device=expression_gt.device)
            # Unnormalize data and predictions
            expression_gt = expression_gt+means
            expression_pred = expression_pred+means

        # Get boolean mask
        mask = torch.Tensor(batch.layers['mask']).to(expression_gt.device).bool()

        return expression_pred, expression_gt, mask

    def training_step(self, batch):

        # Get the patch key in the batch
        # FIXME: Automate this with if self.glob_step == 0: or something simmilar
        patch_key = self.find_batch_patch_key(batch)
        
        # Get (and reshape) images from batch
        tissue_tiles = batch.obsm[patch_key]
        w = round(np.sqrt(tissue_tiles.shape[1]/3))
        tissue_tiles = tissue_tiles.reshape((tissue_tiles.shape[0], w, w, -1))
        
        # Permute dimensions to be in correct order for normalization
        tissue_tiles = tissue_tiles.permute(0,3,1,2).contiguous()
        # Make transformations in tissue tiles
        tissue_tiles = tissue_tiles/255.
        # Transform tiles
        tissue_tiles = self.train_transforms(tissue_tiles)
        
        # Get groundtruth of expression
        expression_gt = batch.X
        # Get boolean mask
        mask = torch.Tensor(batch.layers['mask']).to(expression_gt.device).bool()

        # Get output of the model
        expression_pred = self.forward(tissue_tiles)

        # Compute expression MSE loss (handle case to ignore zeros)
        if self.args.robust_loss == True:
            real_gt, real_pred = expression_gt[mask], expression_pred[mask]
            loss = self.criterion(real_gt, real_pred)
        else:    
            loss = self.criterion(expression_gt, expression_pred)
        
        train_log_dict = {'train_loss': loss}
        self.log_dict(train_log_dict, on_step=True)

        

        # Append train step outputs 
        self.training_step_outputs.append((expression_pred, expression_gt, mask))
        
        return loss

    def on_train_epoch_end(self):
        
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.training_step_outputs)
        # Concatenate outputs along the sample dimension
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put train prefix in metric dict
        metrics = {f'train_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)
        
        # Free memory
        self.training_step_outputs.clear()

    def validation_step(self, batch):
        
        # Get the outputs from the batch with generalistic function
        expression_pred, expression_gt, mask = self.pred_outputs_from_batch(batch)
        # Append validation step outputs 
        self.validation_step_outputs.append((expression_pred, expression_gt, mask))
        
        return expression_pred, expression_gt, mask
    
    def on_validation_epoch_end(self):
        
        if self.trainer.sanity_checking:
            # Free memory
            self.validation_step_outputs.clear() 
            return
        else:
            # Unpack the list of tuples
            glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.validation_step_outputs)
            # Concatenate outputs along the sample dimension
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

    def test_step(self, batch):
        
        # Get the outputs from the batch with generalistic function
        expression_pred, expression_gt, mask = self.pred_outputs_from_batch(batch)
        # Append validation step outputs 
        self.test_step_outputs.append((expression_pred, expression_gt, mask))
        
        return expression_pred, expression_gt, mask

    def on_test_epoch_end(self):
        
        # Unpack the list of tuples
        glob_expression_pred, glob_expression_gt, glob_mask = zip(*self.test_step_outputs)
        # Concatenate outputs along the sample dimension
        glob_expression_pred, glob_expression_gt, glob_mask = torch.cat(glob_expression_pred), torch.cat(glob_expression_gt), torch.cat(glob_mask)

        # Get metrics and log
        metrics = get_metrics(glob_expression_gt, glob_expression_pred, glob_mask)
        # Put test prefix in metric dict
        metrics = {f'test_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)
        
        # Free memory
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        try:
            optimizer = getattr(torch.optim, self.args.optimizer)(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        except:
            optimizer = getattr(torch.optim, self.args.optimizer)(self.parameters(), lr=self.args.lr)

        return optimizer


class EightSymmetry(object):
    """Returns a tuple of the eight symmetries resulting from rotation and reflection.
    
    This behaves similarly to TenCrop.
    This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.
    Example:
     transform = Compose([
         EightSymmetry(), # this is a tuple of PIL Images
         Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
     ])
    """
    # This class function was taken fron the original ST-Net repository at:
    # https://github.com/bryanhe/ST-Net/blob/43022c1cb7de1540d5a74ea2338a12c82491c5ad/stnet/transforms/eight_symmetry.py#L3
    def __call__(self, img):
        identity = lambda x: x
        ans = []
        for i in [identity, RandomHorizontalFlip(1)]:
            for j in [identity, RandomVerticalFlip(1)]:
                for k in [identity, RandomRotation((90, 90))]:
                    ans.append(i(j(k(img))))
        return tuple(ans)

    def __repr__(self):
        return self.__class__.__name__ + "()"