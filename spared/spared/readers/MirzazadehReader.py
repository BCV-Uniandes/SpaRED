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
import wget
from time import time
from datetime import datetime
import json
import zipfile
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # To avoid a warning from opencv
import cv2
import shutil
import tarfile
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

class MirzazadehReader():
    def __init__(self,
        dataset: str = 'mirzazadeh_human_colon',
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
        Mirzazadeh et al. (https://doi.org/10.1038/s41467-023-36071-5) which is hosted in https://data.mendeley.com/datasets/4w6krnywhn/1

        Args:
            dataset (str, optional): An string encoding the dataset name. The following options are available:
                                    'mirzazadeh_human_colon',
                                    'mirzazadeh_human_lung',
                                    'mirzazadeh_human_pediatric_brain_tumor',
                                    'mirzazadeh_human_prostate_cancer',
                                    'mirzazadeh_human_small_intestine',
                                    'mirzazadeh_mouse_bone',
                                    'mirzazadeh_mouse_brain'
                                    Defaults to 'mirzazadeh_human_colon'.
            param_dict (dict, optional): Dictionary that contains filtering and processing parameters. Not used but here for compatibility.
                                        Detailed information about each key can be found in the parser definition over utils.py. 
                                        Defaults to {
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
                                                }.
            patch_scale (float, optional): The scale of the patches to take into account. If bigger than 1, then the patches will be bigger than the original image. Defaults to 1.0.
            patch_size (int, optional): The pixel size of the patches. Patch reshaping is made here. Defaults to 224.
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
        
        # Define train and test data names
        train_data = {
            'mirzazadeh_human_colon_p1':                ['V11M22-349_B1'],
            'mirzazadeh_human_colon_p2':                ['V11B18-363_C1'],
            'mirzazadeh_human_lung':                    ['V10B01-037_A1', 'V10B01-037_B1', 'V10T03-286_B1', 'V10T03-286_D1'],
            'mirzazadeh_human_pediatric_brain_tumor_p1':['V11Y03-081_B1', 'V11Y03-081_C1'],
            'mirzazadeh_human_pediatric_brain_tumor_p2':['V10T03-322_D1'],
            'mirzazadeh_human_prostate_cancer':         ['V11A20-405_C1', 'V11A20-405_D1'],
            'mirzazadeh_human_small_intestine':         ['V19T26-028_B1', 'V19T26-028_D1'],
            'mirzazadeh_mouse_bone':                    ['V11D08-324_A1', 'V11D08-324_C1'],
            'mirzazadeh_mouse_brain':                   ['V10S29-134_A1', 'V10S29-134_B1', 'V10S29-134_C1', 'V10S29-134_D1'],
            'mirzazadeh_mouse_brain_p1':                ['V10S29-134_A1', 'V10S29-134_C1'],
            'mirzazadeh_mouse_brain_p2':                ['V11D08-304_A1', 'V11D08-304_C1']
            }
        
        val_data = {
            'mirzazadeh_human_colon_p1':                ['V11M22-349_A1'],
            'mirzazadeh_human_colon_p2':                ['V10S29-108_B1'],
            'mirzazadeh_human_lung':                    ['V11A20-384_C1', 'V11A20-384_D1'],
            'mirzazadeh_human_pediatric_brain_tumor_p1':['V11Y03-081_A1', 'V11Y03-081_D1'],
            'mirzazadeh_human_pediatric_brain_tumor_p2':['V10T03-322_C1'],
            'mirzazadeh_human_prostate_cancer':         ['V11M22-349_C1', 'V11M22-349_D1'],
            'mirzazadeh_human_small_intestine':         ['V19T26-028_C1'],
            'mirzazadeh_mouse_bone':                    ['V11D08-324_B1'],
            'mirzazadeh_mouse_brain':                   ['V11D08-304_B1', 'V11D08-304_D1'],
            'mirzazadeh_mouse_brain_p1':                ['V10S29-134_B1'],
            'mirzazadeh_mouse_brain_p2':                ['V11D08-304_D1']
            }
        
        test_data = {
            'mirzazadeh_human_colon_p1':                [],
            'mirzazadeh_human_colon_p2':                [],
            'mirzazadeh_human_lung':                    [],
            'mirzazadeh_human_pediatric_brain_tumor_p1':[],
            'mirzazadeh_human_pediatric_brain_tumor_p2':[],
            'mirzazadeh_human_prostate_cancer':         [],
            'mirzazadeh_human_small_intestine':         ['V11B18-363_D1'],
            'mirzazadeh_mouse_bone':                    ['V11D08-324_D1'],
            'mirzazadeh_mouse_brain':                   ['V11D08-304_A1', 'V11D08-304_C1'],
            'mirzazadeh_mouse_brain_p1':                ['V10S29-134_D1'],
            'mirzazadeh_mouse_brain_p2':                ['V11D08-304_B1']
            }
        
        # Get names dictionary
        names_dict = {
            'train':    train_data[self.dataset],
            'val':      val_data[self.dataset],
            'test':     test_data[self.dataset]
            }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict 

    def download_data(self) -> str:
        """
        This function downloads the data of the original Erickson et al. dataset from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4w6krnywhn-1.zip and
        https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/442mhsrpcm-1.zip using wget. Then it unzips the file and deletes the zip file.
        This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # Get dataset name
        dataset_author = self.dataset.split('_')[0]

        # Use wget to download the data
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', f'{dataset_author}_data')) or self.force_compute:
            os.makedirs(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data'), exist_ok=True)
            wget.download('https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4w6krnywhn-1.zip', out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data','zipped_data.zip'))
            wget.download('https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/442mhsrpcm-1.zip', out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data','zipped_data_small_intestine.zip'))
        
            # Unzip the file in a folder with an understandable name
            with zipfile.ZipFile(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'zipped_data.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data'))
            with zipfile.ZipFile(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'zipped_data_small_intestine.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data'))
            
            # Delete the zip files
            os.remove(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'zipped_data.zip'))
            os.remove(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'zipped_data_small_intestine.zip'))
            
            # Avoid the internal folders
            root = os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data')
            children = [os.path.join(root, 'Spatially resolved transcriptomic profiling of degraded and challenging fresh frozen samples Supplementary Data 1'),
                        os.path.join(root, 'Spatially resolved transcriptomic profiling of degraded and challenging fresh frozen samples Supplementary Data 2')]
            for child in children:
                for filename in os.listdir(child):
                    shutil.move(os.path.join(child, filename), os.path.join(root, filename))
                os.rmdir(child)
            
            ### Make specific os commands to organize the data 
            # Move small intestine image folder to images
            shutil.move(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'smallintestine'), os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'images','smallintestine'))
            # Correct miss-placed underscores ('_') in some small intestine images
            for fn in glob.glob(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'images','smallintestine', '*')):
                if len(os.path.basename(fn).split('_'))>2:
                    os.rename(fn, os.path.join(os.path.dirname(fn), os.path.basename(fn).replace('_', '-', 1)))
            # Rename prostate cancer image data
            os.rename(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'images','prostate cancer'), os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'images','prostatecancer'))
            ###

        # Define dictionary mapping datasets to tar file names:
        dataset_2_tarfile = {
            'mirzazadeh_human_colon_p1':                'colon.tar.gz',           
            'mirzazadeh_human_colon_p2':                'colon.tar.gz',           
            'mirzazadeh_human_lung':                    'lung.tar.gz',
            'mirzazadeh_human_pediatric_brain_tumor_p1':'pediatricbraintumor.tar.gz',
            'mirzazadeh_human_pediatric_brain_tumor_p2':'pediatricbraintumor.tar.gz',
            'mirzazadeh_human_prostate_cancer':         'prostatecancer.tar.gz',
            'mirzazadeh_human_small_intestine':         'smallintestine.tar.gz',
            'mirzazadeh_mouse_bone':                    'mousebone.tar.gz',
            'mirzazadeh_mouse_brain':                   'mousebrain.tar.gz',
            'mirzazadeh_mouse_brain_p1':                'mousebrain.tar.gz',
            'mirzazadeh_mouse_brain_p2':                'mousebrain.tar.gz'
        }

        # Create specific dataset folder if it doesn't exist
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', f'{dataset_author}_data', self.dataset)) or self.force_compute:
            
            # Unzip the corresponding tar file into a folder with the dataset name
            with tarfile.open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'spaceranger_output', dataset_2_tarfile[self.dataset]), "r:gz") as file: 
                file.extractall(path=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset))
            
            # Delete intermediary folder
            root = os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset)
            child = os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, dataset_2_tarfile[self.dataset].split('.')[0])
            for filename in os.listdir(child):
                shutil.move(os.path.join(child, filename), os.path.join(root, filename))
            os.rmdir(child)

            # Create empty list of slide ids
            slide_id_list = []

            # Get a list of every current slide id
            for curr_slide_list in self.split_names.values():
                for slide_id in curr_slide_list:
                    slide_id_list.append(slide_id)
            

            # Remove folders in uncompressed data that aren't in slide_id_list
            for dn in glob.glob(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, '*')):
                if os.path.split(dn)[-1] not in slide_id_list:
                    shutil.rmtree(dn)

            # Get paths of each source image using the slide id and move the image to the sample folder
            for slide_id in slide_id_list:
                # Get path of source image
                source_img_path = glob.glob(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', 'images', dataset_2_tarfile[self.dataset].split('.')[0], f'{slide_id}*'))[0]
                # Move source image to corresponding folder
                shutil.copy(source_img_path, os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, 'source_image.jpg'))
            
        return os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset) 

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
            slide_id (str): A string with information of the patient and replicate of the slide. Example: 'V11M22-349_A1'. Where V11M22-349 is the patient
                            and A1 is the replicate.
        
        Returns:
            ad.AnnData: An anndata object with the data of the slide.
        """
        # Get the path for the slice
        path = os.path.join(self.download_path, slide_id)
        # Get the path to the source complete resolution image
        source_img_path = glob.glob(os.path.join(path, '*.jpg'))[0]

        # Read the high resolution image
        hires_img = cv2.imread(os.path.join(path, 'spatial', 'tissue_hires_image.png'))
        # Downsize the hires image to the size of the low resolution image
        low_img = cv2.resize(hires_img, (int((3/10)*hires_img.shape[0]), int((3/10)*hires_img.shape[1])))
        # Save the low resolution image
        cv2.imwrite(os.path.join(path, 'spatial', 'tissue_lowres_image.png'), low_img) 

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
        
        # FIXME: Check where else is this code and remove if unnecessary
        # NOTE: This is a test to fix the problem with the coordinates
        found_img_scale_x = adata.uns['spatial'][sample_name]['images']['hires'].shape[0] / hires_img.shape[0]
        found_img_scale_y = adata.uns['spatial'][sample_name]['images']['hires'].shape[1] / hires_img.shape[1]
        sup_img_scale = adata.uns['spatial'][sample_name]['scalefactors']['tissue_hires_scalef']
        upscale_factor_x = found_img_scale_x / sup_img_scale
        upscale_factor_y = found_img_scale_y / sup_img_scale
        hires_img = cv2.resize(hires_img, (0,0), fx=upscale_factor_x, fy=upscale_factor_y)

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
            x = int(x)
            y = int(y)
            patch = hires_img[y - (window//2):y + (window//2), x - (window//2):x + (window//2), :]
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
        slide_collection.uns = {
            'spatial': {
                slide_id_list[i]: p.uns['spatial'][slide_id_list[i]] for i, p in enumerate(slide_adata_list)
            }
        }
        # Return the patient collection
        return slide_collection