import glob
import anndata as ad
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
import squidpy as sq
import pandas as pd
import numpy as np
from PIL import Image
import warnings
import wget
from time import time
from datetime import datetime
import json
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # To avoid a warning from opencv
import cv2
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

class VicariReader():
    def __init__(self,
        dataset: str = 'vicari_human_striatium', # or 'vicari_mouse_brain'
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
        Abalo et al. (https://doi.org/10.17632/2bh5fchcv6.1) which is hosted in https://data.mendeley.com/datasets/2bh5fchcv6/1

        Args:
            dataset (str, optional): An string encoding the dataset name. The following options are available:
                                    'abalo_human_squamous_cell_carcinoma',
                                    Defaults to 'abalo_human_squamous_cell_carcinoma'.
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
            patch_size (int, optional): The pixel size of the patches. Defaults to 224.
            force_compute (bool, optional): Whether to force the processing computation or not. Not used but here for compatibility. Defaults to False.
        """

        # We define the variables for the SpatialDataset class
        self.dataset = dataset
        self.tissue = "".join(dataset.split("_")[1:])
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
            'vicari_mouse_brain':                    ['V11L12-038_A1', 'V11L12-038_B1', 'V11L12-038_C1', 'V11L12-038_D1',
                                                      'V11L12-109_A1', 'V11L12-109_B1', 'V11L12-109_C1', 'V11L12-109_D1'],
            'vicari_human_striatium':                ['V11T17-102_A1', 'V11T17-102_B1']
            }
        
        val_data = {
            'vicari_mouse_brain':                    ['V11T16-085_A1', 'V11T16-085_B1', 'V11T16-085_C1', 'V11T16-085_D1'],
            'vicari_human_striatium':                ['V11T17-102_C1']
            }
        
        test_data = {
            'vicari_mouse_brain':                    ['V11T17-101_A1', 'V11T17-101_B1'],
            'vicari_human_striatium':                ['V11T17-102_D1']
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
        This function downloads the data of the original Abalo et al. dataset from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/2bh5fchcv6-1.zip
        using wget. Then it unzips the file and deletes the zip file. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # Get dataset name
        dataset_author = self.dataset.split('_')[0]

        links = {
            'V11L12-038_A1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480214',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480217',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480220',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480223',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493688',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480250',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480274'},
            'V11L12-038_B1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480243',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480240',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480249',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480246',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493690',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480265',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480319'},
            'V11L12-038_C1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480311',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480308',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480317',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480314',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493693',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480301',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480358'},
            'V11L12-038_D1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480338',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480335',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480344',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480341',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493691',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480325',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480424'},   
            'V11L12-109_A1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480350',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480347',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480356',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480353',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493697',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480394',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480454'},    
            'V11L12-109_B1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480410',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480407',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480416',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480413',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493699',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480430',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480502'}, 
            'V11L12-109_C1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480446',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480443',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480452',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480449',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493700',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480490',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480799'}, 
            'V11L12-109_D1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480500',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480497',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480782',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480779',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493702',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480679',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480898'},      
            'V11T16-085_A1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480788',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480785',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480794',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480791',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493766',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480883',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40480995'}, 
            'V11T16-085_B1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40480890',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40480887',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40480896',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40480893',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493772',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40480907',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40487627'}, 
            'V11T16-085_C1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40481447',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40484657',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40481456',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40481453',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493778',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40481477',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40481523'}, 
            'V11T16-085_D1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40481621',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40481618',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40481627',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40481624',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493820',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40481538',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40481643'},   
            'V11T17-101_A1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40482452',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40482449',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40482458',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40482455',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493724',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40481685',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40482463'},
            'V11T17-101_B1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40482746',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40482743',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40482752',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40482749',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493733',
                            'fastq_R1': 'https://figshare.scilifelab.se/ndownloader/files/40482721',
                            'fastq_R2': 'https://figshare.scilifelab.se/ndownloader/files/40483708'},
            'V11T17-102_A1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40483314',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40483311',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40483320',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40483317',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493714',
                            'fastq_R1': '',
                            'fastq_R2': ''},
            'V11T17-102_B1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40483416',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40483413',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40483422',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40483419',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493712',
                            'fastq_R1': '',
                            'fastq_R2': ''},
            'V11T17-102_C1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40483587',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40483584',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40483593',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40483590',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493723',
                            'fastq_R1': '',
                            'fastq_R2': ''},
            'V11T17-102_D1': {'json': 'https://figshare.scilifelab.se/ndownloader/files/40483677',
                            'bc_matrix': 'https://figshare.scilifelab.se/ndownloader/files/40483674',
                            'positions': 'https://figshare.scilifelab.se/ndownloader/files/40483686',
                            'hires_image': 'https://figshare.scilifelab.se/ndownloader/files/40483683',
                            'source_image': 'https://figshare.scilifelab.se/ndownloader/files/40493721',
                            'fastq_R1': '',
                            'fastq_R2': ''}       
        }

        # Download the data
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', f'{dataset_author}_data', self.dataset)) or self.force_compute:
            for key, value in self.split_names.items():
                print(f'Downloading {key} data')
                for slide_id in value:
                    os.makedirs(os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, 'spatial'), exist_ok=True)
                    wget.download(links[slide_id]['json'], out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, 'spatial', 'scalefactors_json.json'))
                    wget.download(links[slide_id]['positions'], out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, 'spatial', 'tissue_positions_list.csv'))
                    wget.download(links[slide_id]['hires_image'], out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, 'spatial', 'tissue_hires_image.png'))
                    wget.download(links[slide_id]['bc_matrix'], out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, 'filtered_feature_bc_matrix.h5'))
                    wget.download(links[slide_id]['source_image'], out=os.path.join(SPARED_PATH, 'data', f'{dataset_author}_data', self.dataset, slide_id, f'source_{slide_id}.jpg'))
                                
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
            slide_id (str): A string with information of the patient and replicate of the slide. Example: 'V10F24-015_C1'. Where V10F24-015 is the patient
                            and C1 is the replicate.
        
        Returns:
            ad.AnnData: An anndata object with the data of the slide.
        """
        # Get the path for the slice
        path = os.path.join(self.download_path, slide_id)
        # Get the path to the source complete resolution image
        source_img_path = os.path.join(self.download_path, slide_id, f'source_{slide_id}.jpg')

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

        if 'mouse' in self.dataset.lower():
            old_key = list(adata.uns['spatial'].keys())
            old_key = old_key[0]
            adata.uns['spatial'][slide_id] = adata.uns['spatial'].pop(old_key)

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