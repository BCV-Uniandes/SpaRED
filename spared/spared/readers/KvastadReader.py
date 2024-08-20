import glob
import anndata as ad
import os
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas
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
import gzip
import shutil
import scipy
import matplotlib.pyplot as plt
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

# NOTE: This class is not being currently used nor tested because it is from the ST technology.
class KvastadReader():
    def __init__(self,
        dataset: str = 'kvastad_human_breast_cancer',
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
        Kvastad et al. (https://doi.org/10.1038/s42003-020-01573-1) which is hosted in https://data.mendeley.com/datasets/kzfd6mbnxg/1

        Args:
            dataset (str, optional): An string encoding the dataset type. The following options are available:
                                    'kvastad_human_breast_cancer',
                                    'kvastad_human_childhood_brain_tumor'
                                    Defaults to 'kvastad_mouse_olfactory_bulb'.
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
            'kvastad_human_breast_cancer':          ['BC_P1_D2', 'BC_P1_E1', 'BC_P1_E2'],
            'kvastad_human_childhood_brain_tumor':  ['BT_P1_C1', 'BT_P1_C2', 'BT_P1_E1', 'BT_P1_E2']
            }
        
        val_data = {
            'kvastad_human_breast_cancer':          ['BC_P2_D2', 'BC_P2_E1', 'BC_P2_E2'],
            'kvastad_human_childhood_brain_tumor':  ['BT_P2_C1', 'BT_P1_D1']
            }
        
        test_data = {
            'kvastad_human_breast_cancer':          [],
            'kvastad_human_childhood_brain_tumor':  []
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
        This function downloads the data of the original Kvastad et al. dataset from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/kzfd6mbnxg-1.zip
        using wget. Then it unzips the file and deletes the zip file. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # Use wget to download the data
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', 'kvastad_data', self.dataset)) or self.force_compute:
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'kvastad_data'), exist_ok=True)
            wget.download('https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/kzfd6mbnxg-1.zip', out=os.path.join(SPARED_PATH, 'data', 'kvastad_data','kzfd6mbnxg-1.zip'))
        
            # Unzip the file in a folder with an understandable name
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'kvastad_data', 'unzipped_kvastad_data'))
            with zipfile.ZipFile(os.path.join(SPARED_PATH, "data", "kvastad_data", "kzfd6mbnxg-1.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'kvastad_data', 'unzipped_kvastad_data'))
            
            # Delete the zip file
            os.remove(os.path.join(SPARED_PATH, "data", "kvastad_data", "kzfd6mbnxg-1.zip"))

            # Create two directories in kvastad_data for kvastad_human_breast_cancer and kvastad_human_childhood_brain_tumor
            internal_datasets_names = ['kvastad_human_breast_cancer', 'kvastad_human_childhood_brain_tumor']
            [os.makedirs(os.path.join(SPARED_PATH, 'data', 'kvastad_data', f), exist_ok=True) for f in internal_datasets_names]

            patient_list = ['BC_P1_D2', 'BC_P1_E1', 'BC_P1_E2',
                            'BC_P2_D2', 'BC_P2_E1', 'BC_P2_E2',
                            'BT_P1_C1', 'BT_P1_C2', 'BT_P1_E1', 'BT_P1_E2',
                            'BT_P2_C1', 'BT_P1_D1']
            he_image_list = ['HE_SupFigure7_a_D2.jpg', 'HE_SupFigure7_a_E1.jpg', 'HE_SupFigure7_a_E2.jpg',
                             'HE_SupFigure7_d_D2.jpg', 'HE_SupFigure7_d_E1.jpg', 'HE_SupFigure7_d_E2.jpg',
                             'HE_SupFigure9_d_C1.jpg', 'HE_SupFigure9_d_C2.jpg', 'HE_SupFigure9_d_E1.jpg', 'HE_SupFigure9_d_E2.jpg',
                             'HE_Figure4_c_C1.jpg',	'HE_Figure4_c_D1.jpg']
            spot_coordinates_list = ['spot_coordinates_SupFigure7_b_D2.tsv.gz',	'spot_coordinates_SupFigure7_b_E1.tsv.gz', 'spot_coordinates_SupFigure7_b_E2.tsv.gz',
                                     'spot_coordinates_SupFigure7_e_D2.tsv.gz', 'spot_coordinates_SupFigure7_e_E1.tsv.gz', 'spot_coordinates_SupFigure7_e_E2.tsv.gz',
                                     'spot_coordinates_SupFigure9_a_C1.tsv',	'spot_coordinates_SupFigure9_a_C2.tsv',	   'spot_coordinates_SupFigure9_a_E1.tsv',    'spot_coordinates_SupFigure9_a_E2.tsv',
                                     'spot_coordinates_Figure4_d_C1.tsv',	    'spot_coordinates_Figure4_d_D1.tsv']
            count_matrix_list = ['count_matrix_SupFigure7_b_D2_stdata.tsv.gz', 'count_matrix_SupFigure7_b_E1_stdata.tsv.gz', 'count_matrix_SupFigure7_b_E2_stdata.tsv.gz',
                                 'count_matrix_SupFigure7_e_D2_stdata.tsv.gz', 'count_matrix_SupFigure7_e_E1_stdata.tsv.gz', 'count_matrix_SupFigure7_e_E2_stdata.tsv.gz',
                                 'count_matrix_SupFigure9_a_C1_stdata.tsv.gz', 'count_matrix_SupFigure9_a_C2_stdata.tsv.gz', 'count_matrix_SupFigure9_a_E1_stdata.tsv.gz', 'count_matrix_SupFigure9_a_E2_stdata.tsv.gz',
                                 'count_matrix_Figure4_d_C1_stdata.tsv.gz',    'count_matrix_Figure4_d_D1_stdata.tsv.gz']
            
            # Define manually metadata dataframe
            metadata = pd.DataFrame({
                'histology_image': he_image_list,
                'spot_coordinates': spot_coordinates_list,
                'count_matrix': count_matrix_list},
                index=patient_list)

            # Define simple prefix mapper to use inside the for
            prefix_mapper = {
                'kvastad_human_breast_cancer': 'BC',
                'kvastad_human_childhood_brain_tumor': 'BT'
            }

            # Iterate over internal datasets names to organize all the files in the corresponding folder
            for in_folder in internal_datasets_names:
                
                # Create folders in kvastad_data for count_matrix, histology_image and spot_coordinates
                folder_names = ['count_matrix', 'histology_image', 'spot_coordinates']
                [os.makedirs(os.path.join(SPARED_PATH, 'data', 'kvastad_data', in_folder, f), exist_ok=True) for f in folder_names]

                # Subset metadata to the current internal dataset
                curr_metadata = metadata.loc[metadata.index.str.startswith(prefix_mapper[in_folder])]
                
                # Iterate over the patients in the current internal dataset
                for patient, info in curr_metadata.iterrows():
                    
                    # Move the files to the corresponding folder
                    shutil.move(os.path.join(SPARED_PATH, "data", "kvastad_data", "unzipped_kvastad_data", info['histology_image']), os.path.join(SPARED_PATH, "data", "kvastad_data", in_folder, 'histology_image', f'{patient}_{info["histology_image"]}'))
                    shutil.move(os.path.join(SPARED_PATH, "data", "kvastad_data", "unzipped_kvastad_data", info['spot_coordinates']), os.path.join(SPARED_PATH, "data", "kvastad_data", in_folder, 'spot_coordinates', f'{patient}_{info["spot_coordinates"]}'))
                    shutil.move(os.path.join(SPARED_PATH, "data", "kvastad_data", "unzipped_kvastad_data", info['count_matrix']), os.path.join(SPARED_PATH, "data", "kvastad_data", in_folder, 'count_matrix', f'{patient}_{info["count_matrix"]}'))
            
            # We delete the unzipped folder
            shutil.rmtree(os.path.join(SPARED_PATH, "data", "kvastad_data", "unzipped_kvastad_data"))
            
            # Get the list of all the files in the kvastad_data folder that end in .gz
            gz_files = glob.glob(os.path.join(SPARED_PATH, "data", "kvastad_data", "**", "*.gz"), recursive=True)
            
            # Uncompress all .gz files
            for fn in gz_files:
                with gzip.open(fn, 'rb') as f_in:
                    with open(fn[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Remove all .gz files
            [os.remove(fn) for fn in gz_files]
            
        return os.path.join(SPARED_PATH, 'data', 'kvastad_data', self.dataset) 

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
        No image patch information is added by this function.

        Args:
            slide_id (str): A string with information of the patient and replicate of the slide. Example: 'BC_P1_D2'. Where P1 is the patient
                            and D2 is the replicate. BC is the dataset name.
        
        Returns:
            ad.AnnData: An anndata object with the data of the slide.
        """
        # Get the patient and replicate from the slide name
        patient_id, replicate = slide_id.split('_')[1:]

        # Get the paths to the files to load
        path_dict = {
            'count_matrix': glob.glob(os.path.join(self.download_path, 'count_matrix', f'{slide_id}*.tsv'))[0],
            'spot_coordinates': glob.glob(os.path.join(self.download_path, 'spot_coordinates', f'{slide_id}*.tsv'))[0],
            'histology_image': glob.glob(os.path.join(self.download_path, 'histology_image', f'{slide_id}*.jpg'))[0]
        }
        
        # We load the count matrix, spot coordinates and histology image
        count_matrix = pd.read_csv(path_dict['count_matrix'], index_col = 0, sep='\t', header=0, engine="pyarrow")
        spot_coordinates = pd.read_csv(path_dict['spot_coordinates'], index_col = 0, sep='\t')
        histology_image = plt.imread(path_dict['histology_image'])

        # Handle the index of the spot_coordinates in the human_childhood_brain_tumor dataset that has a slightly different format
        if self.dataset == 'kvastad_human_childhood_brain_tumor':
            spot_coordinates['x'] = spot_coordinates.index
            spot_coordinates.index = spot_coordinates[['x', 'y']].apply(lambda x: f'{x.iloc[0]}x{x.iloc[1]}', axis=1)

        # Standardize the format of the index in the count_matrix and spot_coordinates (format patient_id_replicate_x_y)
        count_matrix.index = [f'{patient_id}_{replicate}_{i.replace("x", "_")}' for i in count_matrix.index]  
        spot_coordinates.index = [f'{patient_id}_{replicate}_{i.replace("x", "_")}' for i in spot_coordinates.index]

        # We compute the intersection between the indexes of the count_matrix and spot_coordinates
        intersection_idx = count_matrix.index.intersection(spot_coordinates.index)

        # Refine count_matrix and spot_coordinates to only contain spots that are in intersection_idx
        count_matrix = count_matrix.loc[intersection_idx]
        spot_coordinates = spot_coordinates.loc[intersection_idx]

        # Handle the fact that the childhod brain tumor dataset has pixel coordinates and that the coordinates of the breast cancer dataset have to be inferred
        if self.dataset == 'kvastad_human_breast_cancer':
            spot_coordinates['pixel_x'] = round(295*spot_coordinates['xcoord'] - 295) # These values were checked with the plots
            spot_coordinates['pixel_y'] = round(289*spot_coordinates['ycoord'] - 289)  # These values were be checked with the plots
        elif self.dataset == 'kvastad_human_childhood_brain_tumor':
            # Define dictionary of constants for each slide_id
            slide_id_2_constants = {
                'BT_P1_C1': {'m_x': 224, 'm_y': 224, 'a_x': 1400, 'a_y':  -30},
                'BT_P1_C2': {'m_x': 220, 'm_y': 218, 'a_x': 1500, 'a_y':  300},
                'BT_P1_E1': {'m_x': 219, 'm_y': 219, 'a_x': 1550, 'a_y':  700},
                'BT_P1_E2': {'m_x': 215, 'm_y': 215, 'a_x': 1600, 'a_y':  650},
                'BT_P2_C1': {'m_x': 153, 'm_y': 157, 'a_x': 1200, 'a_y':  400},
                'BT_P1_D1': {'m_x': 148, 'm_y': 157, 'a_x': 1450, 'a_y': 1300}
            }
            spot_coordinates['ycoord'] = spot_coordinates['y']
            spot_coordinates['xcoord'] = spot_coordinates['x']
            spot_coordinates['pixel_x'] = round(slide_id_2_constants[slide_id]['m_x']*spot_coordinates['xcoord'] + slide_id_2_constants[slide_id]['a_x']) # These values were checked with the plots
            spot_coordinates['pixel_y'] = round(slide_id_2_constants[slide_id]['m_y']*spot_coordinates['ycoord'] + slide_id_2_constants[slide_id]['a_y']) # These values were be checked with the plots
        else:
            raise ValueError(f'The dataset {self.dataset} is not defined for the class KvastadReader.')

        #### Declare obs dataframe
        obs_df = pd.DataFrame({
            'patient': patient_id,
            'replicate': replicate,
            'array_row': spot_coordinates['ycoord'],
            'array_col': spot_coordinates['xcoord'],
        })

        # Add tumor column to obs_df if it is in the spot_coordinates
        if 'tumor' in spot_coordinates.columns:
            obs_df['tumor'] = spot_coordinates['tumor']=='tumor'

        # Set the index name to spot_id
        obs_df.index.name = 'spot_id'

        #### Get the var_df from the count_matrix
        var_df = count_matrix.columns.to_frame()
        var_df.index.name = 'gene_ids'
        var_df.columns = ['gene_ids']
        # Delete characters after the point in the gene_ids and index
        var_df['gene_ids'] = var_df['gene_ids'].str.split('.').str[0]
        var_df.index = var_df.index.str.split('.').str[0]

        #### Declare uns,spatial,sample,metadata dictionary 
        metadata_dict = {
            'chemistry_description': "Spatial Transcriptomics",
            'software_version': 'NA',
            'source_image_path': path_dict['histology_image']
        }

        #### Declare uns,spatial,sample,images dictionary
        # Reshape histology image to lowres (600, 600, 3) and hires (2000, 2000, 3)
        # Read image into PIL
        histology_image = Image.fromarray(histology_image)
        # Resize to lowres
        histology_image_lowres = histology_image.resize((600, int(600*(histology_image.size[1]/histology_image.size[0]))))
        # Resize to hires
        histology_image_hires = histology_image.resize((2000, int(2000*(histology_image.size[1]/histology_image.size[0]))))
        # Convert to numpy array
        histology_image_lowres = np.array(histology_image_lowres)
        histology_image_hires = np.array(histology_image_hires)
        # Create images dictionary
        images_dict = {
            'hires': histology_image_hires,
            'lowres': histology_image_lowres
        }

        # Declare uns,spatial,sample,scalefactors dictionary
        # NOTE: We are trying to compute the scalefactors from the histology image
        # FIXME: Fine tune scale factors for brain tumor dataset the one we have is good for breast cancer
        scalefactors_dict = {
            'fiducial_diameter_fullres': 'NA',
            'spot_diameter_fullres': 291.0, # This diameter was adjusted by making a linear regression in the coordinate files from the childhod brain tumor dataset
            'tissue_hires_scalef': 2000/histology_image.size[0],
            'tissue_lowres_scalef': 600/histology_image.size[0]
        }

        # Declare uns dictionary
        uns_dict = {
            'spatial': {
                slide_id: {
                    'metadata': metadata_dict,
                    'scalefactors': scalefactors_dict,
                    'images': images_dict
                    }

                }
            }

        
        obsm_dict = {
            'spatial': spot_coordinates[['pixel_x', 'pixel_y']].values
        }

        # Declare a scipy sparse matrix from the count matrix
        count_matrix = scipy.sparse.csr_matrix(count_matrix)

        # We create the AnnData object
        adata = ad.AnnData( X = count_matrix,
                            obs = obs_df,
                            var = var_df,
                            obsm = obsm_dict,
                            uns = uns_dict,
                            dtype=np.float32)

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
        In the adata.obs dataframe the columns 'slide_id' and 'split' are added to identify the slide and the split of each observation.
        Also in the var dataframe the column 'exp_frac' is added with the fraction of cells expressing each gene.
        This 'exp_frac' column is the minimum expression fraction of the gene in all the slides.

        Returns:
            ad.AnnCollection: AnnCollection object with all the slides as AnnData objects.
        """

        # Declare patient adata list
        slide_adata_list = []
        slide_id_list = []

        # Iterate over the slide ids of the splits to get the adata for each slide
        print("The first time running this function will take around 3 minutes to read the data in adata format.")
        for key, value in self.split_names.items():
            print(f'Loading {key} data')
            for slide_id in tqdm(value):
                if not os.path.exists(os.path.join(self.download_path,"adata",f'{slide_id}.h5ad')):
                    # Get the adata for the slide
                    adata = self.get_adata_for_slide(slide_id)
                    # Add the patches to the adata
                    adata = self.get_patches(adata)
                    # Add the slide id to a column in the obs
                    adata.obs['slide_id'] = slide_id
                    # Add the key to a column in the obs
                    adata.obs['split'] = key
                    # Add a unique ID column to the observations to be able to track them when in cuda
                    adata.obs['unique_id'] = adata.obs.index
                    # Change the var_names to the gene_ids
                    adata.var_names = adata.var['gene_ids']
                    # Drop the gene_ids column
                    adata.var.drop(columns=['gene_ids'], inplace=True)
                    # Save adata
                    os.makedirs(os.path.join(self.download_path, "adata"), exist_ok=True)
                    adata.write_h5ad(os.path.join(self.download_path, "adata", f"{slide_id}.h5ad"))
                else:
                    # Load adata
                    adata = ad.read_h5ad(os.path.join(self.download_path, "adata", f"{slide_id}.h5ad"))
                    # This overwrites the split column in the adata to be robust to changes in the splits names
                    adata.obs['split'] = key
                    # If the saved adata does not have the patches, then add them and overwrite the saved adata
                    if f'patches_scale_{self.patch_scale}' not in adata.obsm.keys():
                        # Add the patches to the adata
                        adata = self.get_patches(adata)
                        # Save adata
                        adata.write_h5ad(os.path.join(self.download_path, "adata", f"{slide_id}.h5ad"))
                    # Just leave the patches of the scale that is being used. Remove the rest
                    for obsm_key in list(adata.obsm.keys()):
                        if not(obsm_key in [f'patches_scale_{self.patch_scale}', 'spatial']):
                            adata.obsm.pop(obsm_key)

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