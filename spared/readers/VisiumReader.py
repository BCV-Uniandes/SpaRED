import glob
import anndata as ad
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import pandas as pd
import numpy as np
from PIL import Image
import warnings
from time import time
from datetime import datetime
import json
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # To avoid a warning from opencv
import cv2
import tifffile
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

class VisiumReader():
    def __init__(self,
        dataset: str = 'V1_Breast_Cancer_Block_A',
        param_dict: dict = {
            'cell_min_counts':   1000,
            'cell_max_counts':   100000,
            'gene_min_counts':   1e3,
            'gene_max_counts':   1e6,
            'min_exp_frac':      0.8,
            'min_glob_exp_frac': 0.8,
            'top_moran_genes':   256,
            'wildcard_genes':    'None',        
            'combat_key':        'slide_id',    
            'random_samples':    -1,            
            'plotting_slides': 'None',          
            'plotting_genes': 'None',           
                            }, 
        patch_scale: float = 1.0,
        patch_size: int = 224,
        force_compute: bool = False
        ):
        """
        This is a reader class that can download data, get adata objects and compute a collection of slides into an adata object. It is limited to
        reading and will not perform any processing or filtering on the data. In this particular case, it will read the data from Visium datasets.

        Args:
            dataset (str, optional): An string encoding the dataset type. Defaults to 'V1_Breast_Cancer_Block_A'.
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters. Not used but here for compatibility.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
                                                'cell_min_counts':   1000,
                                                'cell_max_counts':   100000,
                                                'gene_min_counts':   1e3,
                                                'gene_max_counts':   1e6, 
                                                'min_exp_frac':      0.8,
                                                'min_glob_exp_frac': 0.8,
                                                'top_moran_genes':   256,
                                                'wildcard_genes':    'None',
                                                'combat_key':        'slide_id',
                                                'random_samples':    -1,
                                                'plotting_slides':   'None',
                                                'plotting_genes':    'None',
                                                }.
            patch_scale (float, optional): The scale of the patches to take into account. If bigger than 1, then the patches will be bigger than the original image. Defaults to 1.0.
            patch_size (int, optional): The pixel size of the patches. Defaults to 224.
            force_compute (bool, optional): Whether to force the processing computation or not. Not used but here for compatibility. Defaults to False.
        """
        # We define the variables for the SpatialDataset class
        self.dataset = dataset
        self.param_dict = param_dict
        self.patch_scale = patch_scale
        self.patch_size = patch_size
        self.force_compute = force_compute
        self.hex_geometry = False if self.dataset == 'stnet_dataset' else True

        # We get the dict of split names
        self.split_names = self.get_split_names()
        # We download the data if it is not already downloaded
        self.download_path = self.download_data()
        # Get the dataset path or create one
        self.dataset_path = self.get_or_save_dataset_path()

    def get_split_names(self) -> dict:
        """
        This function uses the self.dataset variable to return a dictionary of names
        if the data split. The train dataset consists of the first section of the cut.
        While the validation dataset uses the second section. The names are returned in lists
        for compatibility with the STNet dataset class. For this case the test key has no values
        (empty list).

        Returns:
            dict: Dictionary of data names for train, validation and test in lists.
        """
        # Define train and test data names
        train_data = {
            "10xgenomic_mouse_brain_sagittal_anterior":     "V1_Mouse_Brain_Sagittal_Anterior",
            "10xgenomic_mouse_brain_coronal":               "V1_Adult_Mouse_Brain_Coronal_Section_1",
            "10xgenomic_human_breast_cancer":               "V1_Breast_Cancer_Block_A_Section_1",
            "10xgenomic_mouse_brain_sagittal_posterior":    "V1_Mouse_Brain_Sagittal_Posterior",
            "10xgenomic_human_brain":                       "V1_Human_Brain_Section_1"
            }
        
        val_data = {
            "10xgenomic_mouse_brain_sagittal_anterior":     "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
            "10xgenomic_mouse_brain_coronal":               "V1_Adult_Mouse_Brain_Coronal_Section_2",
            "10xgenomic_human_breast_cancer":               "V1_Breast_Cancer_Block_A_Section_2",
            "10xgenomic_mouse_brain_sagittal_posterior":    "V1_Mouse_Brain_Sagittal_Posterior_Section_2",
            "10xgenomic_human_brain":                       "V1_Human_Brain_Section_2"
            }
        
        # Get names dictionary
        names_dict = {
            'train': [train_data[self.dataset]],
            'val': [val_data[self.dataset]],
            'test': [] 
            }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict

    def download_data(self) -> str:
        """
        This function downloads the visium data specified and downloads it into
        os.path.join(SPARED_PATH, 'data', 'Visium_data', self.dataset).
        This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # If the path of the data does not exist, then download the data
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', 'Visium_data', self.dataset)) or self.force_compute:
            
            # Iterate over all the names of the split data and download them
            for sample_list in self.split_names.values():
                for element_id in sample_list:
                    _ = sq.datasets.visium(
                        sample_id = element_id,
                        base_dir = os.path.join(SPARED_PATH, 'data', 'Visium_data', self.dataset),
                        include_hires_tiff = True
                        )
        
        return os.path.join(SPARED_PATH, 'data', 'Visium_data', self.dataset)

    def get_or_save_dataset_path(self) -> str:
        """
        This function saves the parameters of the dataset in a dictionary on a path in the
        processed_dataset folder. The path is returned.

        Returns:
            str: Path to the saved parameters.
        """

        # Get all the class attributes of the current dataset
        curr_dict = self.__dict__.copy()
        
        # Delete some keys from dictionary in order to just leave the class parameters
        curr_dict.pop('download_path', None)
        curr_dict.pop('force_compute', None)
        curr_dict.pop('plotting_genes', None)
        curr_dict.pop('plotting_slides', None)

        # Define parent folder of all saved datasets
        parent_folder = self.download_path.replace('data', 'processed_data', 1)

        # Get the filenames of the parameters of all directories in the parent folder
        filenames = glob.glob(os.path.join(parent_folder, '**', 'parameters.json'), recursive=True)

        # Iterate over all the filenames and check if the parameters are the same
        for filename in filenames:
            with open(filename, 'r') as f:
                # Load the parameters of the dataset
                saved_params = json.load(f)
                # Check if the parameters are the same
                if saved_params == curr_dict:
                    print(f'Parameters already saved in {filename}')
                    return os.path.dirname(filename)

        # If the parameters are not saved, then save them
        # Define directory path to save data
        save_path = os.path.join(parent_folder, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        # Make directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        # Save json
        with open(os.path.join(SPARED_PATH, save_path, 'parameters.json'), 'w') as f:
            json.dump(curr_dict, f, sort_keys=True, indent=4)


        print(f'Parameters not found so this set of parameters is saved in {save_path}')

        return save_path

    def get_adata_for_slide(self, slide_id: str) -> ad.AnnData:
        """
        This function receives a slide id and returns an adata object by reading the previously downloaded data.

        Args:
            slide_id (str): The id of the slice to read. E.g. V1_Breast_Cancer_Block_A_Section_2

        Returns:
            ad.AnnData: An anndata object with the data of the slice.
        """
        # Get the path for the slice
        path = os.path.join(self.download_path, slide_id)
        # Read the data
        adata = sq.read.visium(path, gex_only=False)
        # Add the path to the original image to the metadata
        adata.uns['spatial'][f'{slide_id}']['metadata']['source_image_path'] = os.path.join(path, 'image.tif')
        return adata
    
    def get_patches(self, adata: ad.AnnData) -> ad.AnnData:
        """
        This function gets the image patches around a sample center accordingly to a defined scale and then adds them to an observation metadata matrix called 
        adata.obsm[f'patches_scale_{self.patch_scale}'] in the original anndata object. The added matrix has as rows each observation and in each column a pixel of the flattened
        patch.

        Args:
            adata (ad.AnnData): Original anndata object to get the parches. Must have the route to the super high resolution image.
        
        Returns:
            ad.AnnData: An anndata object with the flat patches added to the observation metadata matrix adata.obsm[f'patches_scale_{self.patch_scale}'].
        """
        # Get the name of the sample
        sample_name = list(adata.uns['spatial'].keys())[0]
        # Get the path and read the super high resolution image
        hires_img_path = adata.uns['spatial'][sample_name]['metadata']["source_image_path"]

        # Check the image dimensions are in the correct order
        hires_img = tifffile.imread(hires_img_path)
        if hires_img.shape[0] < hires_img.shape[1] and hires_img.shape[0] < hires_img.shape[2]:
            hires_img = np.transpose(hires_img, (1, 2, 0))
        elif hires_img.shape[1] < hires_img.shape[0] and hires_img.shape[1] < hires_img.shape[2]:
            hires_img = np.transpose(hires_img, (0, 2, 1))

        # Verify the image only has 3 channels
        if hires_img.shape[-1] != 3:
             hires_img = hires_img[:, :, :-1]
        
        # Get the spatial coordinates of the centers of the spots
        coord =  pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)

        # Get the size of the window to get the patches
        org_window = int(adata.uns['spatial'][sample_name]['scalefactors']['spot_diameter_fullres']) 
        window = int(org_window * self.patch_scale)

        # If the window is odd, then add one to make it even
        if window % 2 == 1:
            window += 1
        
        # If the window is bigger than the original window, then the image must be padded
        if window > org_window:
            # Get the difference between the original window and the new window
            diff = window - org_window
            # Pad the image
            hires_img = np.pad(hires_img, ((diff//2, diff//2), (diff//2, diff//2), (0, 0)), mode='symmetric')
            # Update the coordinates to the new padded image
            coord['x_coord'] = coord['x_coord'] + diff//2
            coord['y_coord'] = coord['y_coord'] + diff//2

        # Define zeros matrix to store the patches
        flat_patches = np.zeros((adata.n_obs, self.patch_size**2*hires_img.shape[-1]), dtype=np.uint8)

        # Iterate over the coordinates and get the patches
        for i, (x, y) in enumerate(coord.values):
            # Get the patch
            patch = hires_img[y - (window//2):y + (window//2), x - (window//2):x + (window//2), :]
            if self.dataset == "10xgenomic_mouse_brain_coronal" or self.dataset == "10xgenomic_human_brain":
                # Normalize the image to [0, 1] range
                normalized_patch = patch.astype(np.float32) / 65535.0
                # Scale the normalized image to [0, 255] range
                patch = (normalized_patch * 255).astype(np.uint8)
            # Reshape patch to desired size
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
            # Flatten the patch
            flat_patches[i,:] = patch.flatten()

        # Add the flat crop matrix to a layer in a data
        adata.obsm[f'patches_scale_{self.patch_scale}'] = flat_patches

        return adata    

    def get_adata_collection(self) -> ad.AnnData:
        """
        This function reads all the adata objects for the slides in the splits and returns a concatenated AnnData object with all the slides.
        In the adata.obs dataframe the columns 'slide_id' and 'split' are added to identify the slide and the split of each observation.

        Returns:
            ad.AnnCollection: AnnCollection object with all the slides as AnnData objects.
        """

        # Declare patient adata list
        slide_adata_list = []
        slide_id_list = []

        # Iterate over the slide ids of the splits to get the adata for each slide
        for key, value in self.split_names.items():
            print(f'Loading {key} data')
            for slide_id in value:
                # Get the adata for the slice
                adata = self.get_adata_for_slide(slide_id)
                # Add the patches to the adata
                adata = self.get_patches(adata)
                # Add the slide id as a prefix to the obs names
                adata.obs_names = [f'{slide_id}_{obs_name}' for obs_name in adata.obs_names]
                # Add the slide id to a column in the obs
                adata.obs['slide_id'] = slide_id
                # Add the key to a column in the obs
                adata.obs['split'] = key
                # Add the patient key to a column in the obs
                adata.obs['patient'] = "unique"
                # Add a unique ID column to the observations to be able to track them when in cuda
                adata.obs['unique_id'] = adata.obs.index
                # Change the var_names to the gene_ids
                adata.var_names = adata.var['gene_ids']
                # Add the adata to the list
                slide_adata_list.append(adata)
                # Add the slide id to the list
                slide_id_list.append(slide_id)
        
        # Concatenate all the patients in a single AnnCollection object
        slide_collection = ad.concat(
            slide_adata_list,
            join='inner',
            merge='same'
        )

        # Define a uns dictionary of the collection
        slide_collection.uns = {
            'spatial': {
                slide_id_list[i]: p.uns['spatial'][slide_id_list[i]] for i, p in enumerate(slide_adata_list)
            }
        }

        # Return the patient collection
        return slide_collection