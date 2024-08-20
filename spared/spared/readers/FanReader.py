import glob
import anndata as ad
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import warnings
from time import time
from datetime import datetime
import json
import zipfile
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # To avoid a warning from opencv
import cv2
import shutil
from urllib import request
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]


class FanReader():
    def __init__(self,
        dataset: str = 'fan_mouse_brain_coronal', 
        param_dict: dict = {
            'cell_min_counts':   500,
            'cell_max_counts':   100000,
            'gene_min_counts':   1e3,
            'gene_max_counts':   1e6,
            'min_exp_frac':      0.2,
            'min_glob_exp_frac': 0.6,
            'top_moran_genes':   256,
            'wildcard_genes':    'None',
            'combat_key':        'patient',       
            'random_samples':    -1,              
            'plotting_slides':   'None',          
            'plotting_genes':    'None',          
            }, 
        patch_scale: float = 1.0,
        patch_size: int = 224,
        force_compute: bool = False
        ):
        """
        This is a reader class that can download data, get adata objects and compute a collection of slides into an adata object. It is limited to
        reading and will not perform any processing or filtering on the data. In this particular case, it will read the data from the dataset published by
        Fan et al. (https://doi.org/10.1101/2022.10.25.513696) which is hosted in https://data.mendeley.com/datasets/nrbsxrk9mp/1

        Args:
            dataset (str, optional): An string encoding the dataset name. 
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters. Not used but here for compatibility.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
                                                'cell_min_counts':      1000,
                                                'cell_max_counts':      100000,
                                                'gene_min_counts':      1e3,
                                                'gene_max_counts':      1e6, 
                                                'min_exp_frac':         0.8,
                                                'min_glob_exp_frac':    0.8,
                                                'real_data_percentage': 0.7,
                                                'top_moran_genes':      256,
                                                'wildcard_genes':       'None',
                                                'combat_key':           'slide_id',
                                                'random_samples':       -1,
                                                'plotting_slides':      'None',
                                                'plotting_genes':       'None',
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
        self.hex_geometry = False

        # We get the dict of split names
        self.split_names = self.get_split_names()
        # We download the data if it is not already downloaded
        self.download_path = self.download_data()
        # Get the dataset path or create one
        self.dataset_path = self.get_or_save_dataset_path()

    def get_split_names(self) -> dict:
        """
        This function uses the self.dataset variable to return a dictionary of names
        if the data split. 
        Returns:
            dict: Dictionary of data names for train, validation and test in lists.
        """
        #Define train, val and test data for each dataset
        train_data = {
            'fan_mouse_brain_coronal': ['HC_B1'],
            'fan_mouse_olfatory_bulb': ['MOB_D1']
        }

        val_data = {
            'fan_mouse_brain_coronal': ['HC_A1'],
            'fan_mouse_olfatory_bulb': ['MOB_C1']
        }

        test_data = {
            'fan_mouse_brain_coronal': [],
            'fan_mouse_olfatory_bulb': []
        }
        
        # Get names dictionary
        names_dict = {
            'train':  train_data[self.dataset],
            'val': val_data[self.dataset],
            'test': test_data[self.dataset]
            }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict 
    
    def download_data(self) -> str:
        """
        This function downloads all the Visium data related with this paper Fan et al. (https://doi.org/10.1101/2022.10.25.513696)
        using wget to the data/fan_mouse_brain_data directory. Then it unzips the files and deletes the zip files. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory with the images and count_matrix
        """
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', 'fan_mouse_brain_data')) or self.force_compute:
            #Create the folder for this dataset and download the original .zip folder with all the data
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data'), exist_ok=True)
            # Define the remote file to retrieve
            remote_url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/nrbsxrk9mp-1.zip'
            # Define the local filename to save data
            local_file = os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'zipped_data.zip')
            print("\n Downloading .zip files takes approximately 2 minutes ")
            # Download remote and save locally
            request.urlretrieve(remote_url, local_file)
            print("\n Download is complete")
            #Unzip the folder
            with zipfile.ZipFile(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'zipped_data.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data'))

            # Delete the zip file
            os.remove(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'zipped_data.zip'))

            #Delete unnecesary files in the unzip folder
            shutil.rmtree(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data','Ex-ST', 'processed_data', 'stereoscope_output_expST_standard_Visium_MOB_hippocampus'))
            shutil.rmtree(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data','Ex-ST', 'processed_data', 'stereoscope_output_modified_visium'))

            #Delete folder that doesnt have raw image
            shutil.rmtree(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Ex-ST', 'processed_data', 'spaceranger_output', 'hippocampus_standard_Visium_10XGenomics'))
            
            #Move images to a new folder 
            images_files = os.listdir(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Ex-ST', 'raw_imgs'))
            #Create a new folder for the full resolution images
            os.mkdir(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'images_FR'))
            for image in images_files:
                origen = os.path.join(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Ex-ST', 'raw_imgs'), image)
                destino = os.path.join(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'images_FR'), image)
                shutil.move(origen, destino)

            # Move Visium folders to a new folder
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Visium_Folders'), exist_ok=True)
            #List with the names of the folders
            carpeta_padre = os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Ex-ST', 'processed_data', 'spaceranger_output')
            folders = [nombre for nombre in os.listdir(carpeta_padre) if os.path.isdir(os.path.join(carpeta_padre, nombre))]

            for folder in folders:
                origen = os.path.join(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Ex-ST', 'processed_data', 'spaceranger_output'), folder)
                destino = os.path.join(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Visium_Folders'), folder)
                shutil.move(origen, destino)

            #Delete unneccesary folder
            shutil.rmtree(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Ex-ST'))
            
            #Create folder for fan_mouse_brain_coronal dataset
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_brain_coronal'), exist_ok=True)
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_brain_coronal', 'images_FR'), exist_ok=True)
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_brain_coronal', 'Visium_Folders'), exist_ok=True)

            #Create folder for fan_mouse_olfatory_bulb dataset
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_olfatory_bulb'), exist_ok=True)
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_olfatory_bulb', 'images_FR'), exist_ok=True)
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_olfatory_bulb', 'Visium_Folders'), exist_ok=True)

            #Change names of the files to the format patient_slice
            original_images_folder = glob.glob(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'images_FR', '*.jpg'))
            
            for path in original_images_folder:
                name = path.split("/")[-1] 
                #Change names and paths for the images of the dataset
                if 'modif' in name:
                    new_name = name.split('_')[-2:]
                    if 'HC' in name:
                        shutil.move(path, os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_brain_coronal', 'images_FR', new_name[0][:-1] + "_" + new_name[1]))
                    elif 'MOB' in name:
                        shutil.move(path, os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_olfatory_bulb', 'images_FR', new_name[0][:-1] + "_" + new_name[1]))

            # Change the name of the Visium folders
            original_Visium_folder = os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Visium_Folders')
            Visium_folders = os.listdir(original_Visium_folder)

            #Helper dictionary for the names
            slices_names = {'HC1': 'A1', 'HC2': 'B1', 'MOB1': 'C1', 'MOB2': 'D1'}
            for folder in Visium_folders:
                if 'modif' in folder:
                    new_name = folder.split('_')[2][:-1] + '_' + slices_names[folder.split('_')[2]]
                    if 'HC' in folder:
                        shutil.move(os.path.join(original_Visium_folder, folder), 
                                    os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_brain_coronal', 'Visium_Folders', new_name))
                    elif 'MOB' in folder:
                        shutil.move(os.path.join(original_Visium_folder, folder), 
                                 os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'fan_mouse_olfatory_bulb', 'Visium_Folders', new_name))
                        
            # Delete unnecesary folder
            shutil.rmtree(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'images_FR'))
            shutil.rmtree(os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', 'Visium_Folders'))

        return os.path.join(SPARED_PATH, 'data', 'fan_mouse_brain_data', self.dataset)

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
        with open(os.path.join(save_path, 'parameters.json'), 'w') as f:
            json.dump(curr_dict, f, sort_keys=True, indent=4)
        print(f'Parameters not found so this set of parameters is saved in {save_path}')
        return save_path

    def get_adata_for_slide(self, slide_id: str) -> ad.AnnData:
        """
        This function loads the data from the given slide_id and replicate and returns an AnnData object all the with relevant information.
        No image patch information is added by this function. The var names in the adata object are gene_ids in ensembl format without
        version specification. E.g. ENSG00000243485

        Args:
            slide_id (str): A string with information of the patient and replicate of the slide. Example: 'expHB_1'. 
        
        Returns:
            ad.AnnData: An anndata object with the data of the slide.
        """
        # Get the path for the slice
        path = os.path.join(self.download_path, 'Visium_Folders' ,slide_id)
        # Get the path to the source complete resolution image
        carpeta_padre = os.path.join(self.download_path, 'images_FR')
        nombres_carpetas = glob.glob(os.path.join(carpeta_padre, '*.jpg'))
        for nombre in nombres_carpetas:
            if slide_id in nombre:
                source_img_path = nombre
        
        # Read the data
        adata = sq.read.visium(path, source_image_path=source_img_path)
        # Add current var names as gene_symbol column
        adata.var['gene_symbol'] = adata.var_names
        # Change the var_names to the gene_ids
        adata.var_names = adata.var['gene_ids']

        # Add a patient column to the obs
        adata.obs['patient'] = slide_id.split('_')[0]

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

        # Read the full hires image into numpy array
        hires_img = cv2.imread(hires_img_path)
        # Pass from BGR to RGB
        hires_img = cv2.cvtColor(hires_img, cv2.COLOR_BGR2RGB)
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
        #flat_patches = np.zeros((adata.n_obs, window**2*hires_img.shape[-1]), dtype=np.uint8)

        flat_patches = np.zeros((adata.n_obs, self.patch_size**2*hires_img.shape[-1]), dtype=np.uint8)
        # Iterate over the coordinates and get the patches
        for i, (x, y) in enumerate(coord.values):
            # Get the patch
            x = int(x)
            y = int(y)
            patch = hires_img[y - (window//2):y + (window//2), x - (window//2):x + (window//2), :]
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
            # Flatten the patch
            flat_patches[i,:] = patch.flatten()

        # Add the flat crop matrix to a layer in a data
        adata.obsm[f'patches_scale_{self.patch_scale}'] = flat_patches

        return adata

    def get_adata_collection(self) -> ad.AnnData:
        """
        This function reads all the adata objects for the slides in the splits and returns a concatenated AnnData object with all the slides.
        In the adata.obs dataframe the columns 'slide_id', 'unique_id', and 'split' are added to identify the slide, each observation and the
        split of each observation.

        Returns:
            ad.AnnCollection: AnnCollection object with all the slides as AnnData objects.
        """

        # Declare patient adata list
        slide_adata_list = []
        slide_id_list = []

        # Iterate over the slide ids of the splits to get the adata for each slide
        for key, value in self.split_names.items():
            print(f'Loading {key} data')
            for slide_id in tqdm(value):
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
                # Add a unique ID column to the observations to be able to track them when in cuda
                adata.obs['unique_id'] = adata.obs.index
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
        real_values = {'expHB_1':'V10A13-160_B1', 
                        'expHB_2': 'V10A13-160_A1',
                        'expMOB_1': 'V10A13-160_C1',
                        'expMOB_2': 'V10A13-160_D1',
                        'HC_A1': 'V10A13-160_A1',
                        'HC_B1': 'V10A13-160_B1',
                        'MOB_C1': 'V10A13-160_C1',
                        'MOB_D1': 'V10A13-160_D1'
                        }
        slide_collection.uns = {
            'spatial': {
                slide_id_list[i]: p.uns['spatial'][real_values[slide_id_list[i]]] for i, p in enumerate(slide_adata_list)
            }
        }
        # Return the patient collection
        return slide_collection
