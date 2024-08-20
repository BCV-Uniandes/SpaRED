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
import itertools 
from sh import gunzip
import h5py
import tifffile
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import requests

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

class HudsonReader():
    def __init__(self,
        dataset: str = 'hudson_human_brain',
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
        Hudson et al. (https://pubmed.ncbi.nlm.nih.gov/35707680/) which is hosted in https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE179572
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
        # Get names dictionary
        names_dict = {
            'train': ["GSM5420749_0", "GSM5420751_0", "GSM5420752_0", "GSM5420754_0"],
            'val':   ["GSM5420750_0"],
            'test':  ["GSM5420753_0"] 
        }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict 

    def download_data(self) -> str:
        """
        This function downloads the data of the original Hudson et al. dataset from (https://pubmed.ncbi.nlm.nih.gov/35707680/) and
        https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE179572 using wget. Then it unzips the file and deletes the zip file.
        This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # Get dataset name
        dataset_author = self.dataset.split('_')[0]

        # Use wget to download the data
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', f'{dataset_author}_data')) or self.force_compute:
            os.makedirs(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data'), exist_ok=True)
            # Download all spatial images
            response = requests.get("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE179nnn/GSE179572/suppl/GSE179572%5Fall%5Fspatial%5Fimages.tar.gz")
            with open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data','GSE179572_all_spatial_images.tar.gz'), 'wb') as file:
                    file.write(response.content)
            # Unzip tar file all_spatial_images
            with tarfile.open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', "GSE179572_all_spatial_images.tar.gz"), "r:gz") as file: 
                file.extractall(path=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data'))
            
            # Remove all_spatial_images zip file   
            os.remove(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', "GSE179572_all_spatial_images.tar.gz"))

            # Download files from each patient 
            patients = ["GSM5420749",  "GSM5420750",  "GSM5420751",  "GSM5420752",  "GSM5420753",  "GSM5420754"]
            pt_number = ["15", "16", "19", "24", "26", "27"]
            
            for patient, pt in zip(patients, pt_number):
                os.makedirs(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0"), exist_ok=True)
                #Download all files from all the patients
                response = requests.get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5420nnn/{patient}/suppl/{patient}%5Fspatial.tar.gz")
                with open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_spatial.tar.gz"), 'wb') as file:
                    file.write(response.content)   
                response = requests.get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5420nnn/{patient}/suppl/{patient}%5Fpt{pt}%5Fmatrix.mtx.gz")
                with open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_matrix.mtx.gz"), 'wb') as file:
                    file.write(response.content)  
                response = requests.get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5420nnn/{patient}/suppl/{patient}%5Fpt{pt}%5Fbarcodes.tsv.gz")
                with open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_barcodes.tsv.gz"), 'wb') as file:
                    file.write(response.content)  
                response = requests.get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5420nnn/{patient}/suppl/{patient}%5Fpt{pt}%5Ffeatures.tsv.gz")
                with open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_features.tsv.gz"), 'wb') as file:
                    file.write(response.content)  
                response = requests.get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5420nnn/{patient}/suppl/{patient}%5Ffiltered%5Ffeature%5Fbc%5Fmatrix.h5")
                with open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_filtered_feature_bc_matrix.h5"), 'wb') as file:
                    file.write(response.content)
                    
            # Unzip de gz and tar files
            for patient in tqdm(patients):
                with tarfile.open(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_spatial.tar.gz"), "r:gz") as file: 
                    file.extractall(path=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0"))
                os.remove(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_spatial.tar.gz"))
                
                gunzip(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_matrix.mtx.gz"))
                gunzip(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_barcodes.tsv.gz"))
                gunzip(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', f'{dataset_author}_human_brain', f"{patient}_0", f"{patient}_features.tsv.gz"))                          
    
            # Modify source image .tif to be alinged with image aligned_fiducials
            for count, pt in enumerate(pt_number):
                input_image_path = f"/media/user_home0/dvegaa/SEPAL/spared/data/hudson_data/all_spatial_images/pt{pt}.tif"
                tif_image = tifffile.imread(input_image_path)
                output_image_path = f"/media/user_home0/dvegaa/SEPAL/spared/data/hudson_data/hudson_human_brain/{patients[count]}_0/pt{pt}.tif"
                
                if (pt == "16") or (pt == "24") or (pt == "26") or (pt == "27"):
                    tif_image = np.rot90(tif_image)
                    tif_image = np.fliplr(tif_image)             
                
                tifffile.imsave(output_image_path, tif_image)
            
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
        version specification. E.g. ENSG00000277196

        Args:
            slide_id (str): A string with information of the patient and replicate of the slide. Example: 'GSM5420749_0'. Where GSM5420749 is the patient
                            and 0 is the replicate.
        
        Returns:
            ad.AnnData: An anndata object with the data of the slide.
        """
        # Get the path for the slice
        path = os.path.join(self.download_path, slide_id)
        # Get the path to the source complete resolution image
        source_img_path = glob.glob(os.path.join(path, '*.tif'))[0]
        # Read the high resolution image
        hires_img = cv2.imread(os.path.join(path, 'spatial', 'tissue_hires_image.png'))
        # Downsize the hires image to the size of the low resolution image
        low_img = cv2.imread(os.path.join(path, 'spatial', 'tissue_lowres_image.png'))
        # Read the data
        adata = sq.read.visium(path, counts_file= f"{slide_id.split('_')[0]}_filtered_feature_bc_matrix.h5", source_image_path=source_img_path)
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
        hires_img = tifffile.imread(hires_img_path)
        # Pass from BGR to RGB
        hires_img = cv2.cvtColor(hires_img, cv2.COLOR_BGR2RGB)
        # Get the spatial coordinates of the centers of the spots
        coord =  pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index=adata.obs_names)

        """
        #NOTE this is a test to verify correct alignement of images and coordinates
        plt.figure()
        img = plt.imread(hires_img_path)
        plt.imshow(img)
        plt.scatter(coord["x_coord"], coord["y_coord"], color = "red")
        out = hires_img_path.split(".")[0]
        output_path = out + "_coords.jpg"
        plt.savefig(output_path)
        """
        
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
                # Add the slide id to the list (change format to patient_0)
                slide_id_list.append(slide_id)
        
        # Concatenate all the patients in a single AnnCollection object
        slide_collection = ad.concat(
            slide_adata_list,
            join='inner',
            merge='same'
        )
        # Define a uns dictionary of the collection
        # Define uns_ids for every slide_id (uns_ids are different from slide_ids)
        uns_ids = {"GSM5420749_0": "sample3", "GSM5420750_0": "L4", "GSM5420751_0": "sample4", "GSM5420752_0": "L3", "GSM5420753_0": "L1", "GSM5420754_0": "L2"}
        slide_collection.uns = {
            'spatial': {
                slide_id_list[i]: p.uns['spatial'][uns_ids[slide_id_list[i]]] for i, p in enumerate(slide_adata_list)
            }
        }
        # Return the patient collection
        return slide_collection