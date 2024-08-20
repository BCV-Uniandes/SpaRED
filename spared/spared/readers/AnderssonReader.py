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
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # To avoid a warning from opencv
import cv2
import gzip
import scipy
import pyzipper
import shutil
from matplotlib import pyplot as plt
import pathlib


# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None
# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

# NOTE: This class is not being currently used nor tested because it is from the ST technology.
class AnderssonReader():
    def __init__(self,
        dataset: str = "andersson_human_breast_cancer_dataset",
        #REVISAR
        param_dict: dict = {
            'cell_min_counts':      500,
            'cell_max_counts':      100000,
            'gene_min_counts':      1e3,
            'gene_max_counts':      1e6,
            'min_exp_frac':         0.2,
            'min_glob_exp_frac':    0.6,
            'real_data_percentage': 0.7,
            'top_moran_genes':      256,
            'wildcard_genes':       'None',
            'combat_key':           'patient',       
            'random_samples':       -1,              
            'plotting_slides':        'None',          
            'plotting_genes':          'None',
            },
        patch_scale: float = 1.0,
        patch_size: int = 294,
        force_compute: bool = False,
        ):
        """
        This is a reader class that can download the data from Andersson dataset.
        Args: 
        """
        #We define the attributes for the HER2+ dataset
        self.dataset = dataset
        self.param_dict = param_dict
        self.patch_scale = patch_scale
        self.patch_size = patch_size
        self.force_compute = force_compute
        self.hex_geometry = False


        # We download the data if it is not already downloaded
        self.download_path = self.download_data()
        # We get the dict of split names
        self.split_names = self.get_split_names()
        # Get the dataset path or create one
        self.dataset_path = self.get_or_save_dataset_path()

    def get_split_names(self) -> dict:
        """
        This function uses the self.dataset variable to return a dictionary of names
        if the data split. 
        Returns:
            dict: Dictionary of data names for train, validation and test in lists.
        """
        # Get names dictionary
        names_dict = {
            'train': ["A1", "A2", "A3", "A4", "A5", "A6",
                      "B1", "B2", "B3", "B4", "B5", "B6",
                      "C1", "C2", "C3", "C4", "C5", "C6",
                      "D1", "D2", "D3", "D4", "D5", "D6"],
            'val':   ["E1", "E2", "E3", "F1", "F2", "F3"],
            'test':  ["G1", "G2", "G3", "H1", "H2", "H3"] 
        }

        
        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')
        
        #Print dataset stadistics
        for key, value in names_dict.items():
            print(f'Number of slices in {key} data: {len(value)}')
        return names_dict 
    
    def download_data(self) -> str:
        """
        This function downloads all the data of the original HER2+ dataset from https://zenodo.org/record/4751624
        using wget to the data/HER2_data directory. Then it unzips the files and deletes the zip files. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', 'andersson_human_breast_cancer_data')) or self.force_compute:
            print("The download process of the HER2+ dataset takes approximately 10 - 11 min.")

            #Use wget to download the images
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data'), exist_ok=True)
            wget.download('https://zenodo.org/record/4751624/files/images.zip?download=1', out=os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_images.zip"))
            
            #Unzip the images and save it in other folder
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'unzipped_HER2_images'), exist_ok=True)
            with pyzipper.AESZipFile(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_images.zip"), 'r', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zip_ref:
                try:
                    zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"unzipped_HER2_images"), pwd=str.encode('zNLXkYk3Q9znUseS'))
                except: 
                    pass

            # There is an extra folder inside the unzipped folder of images. We move the files to a new folder.
            os.makedirs(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "histology_image"), exist_ok = True)
            files = os.listdir(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_images", "images", "HE"))
            for file in files:
                shutil.move(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_images", "images", "HE", file), os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "histology_image"))

            # Move the annotation folder inside the unzipped_HER2_images
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'meta'), exist_ok = True)
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'meta', 'annotation'), exist_ok = True)
            annotation_files = glob.glob(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_images", "images", "annotation", "*.jpg"))
            for file in annotation_files:
                file_name = os.path.basename(file)
                shutil.move(file, os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "meta", "annotation" , file_name))
            

            # Delete the zip file
            os.remove(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_images.zip"))

            #Delete the unzipper_HER2_images folder
            shutil.rmtree(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_images"))

            #Use wget to download the count-matrices
            wget.download('https://zenodo.org/record/4751624/files/count-matrices.zip?download=1', out=os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_count_matrices.zip"))
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'unzipped_HER2_count_matrices'), exist_ok=True)
            with pyzipper.AESZipFile(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_count_matrices.zip"), 'r', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zip_ref:
                zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"unzipped_HER2_count_matrices"), pwd=str.encode('zNLXkYk3Q9znUseS'))
                
            # There is an extra folder inside the unzipped folder. We move the files to the unzipped folder.
            files = os.listdir(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_count_matrices", "count-matrices"))
            for file in files:
                shutil.move(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_count_matrices", "count-matrices",file),os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_count_matrices"))

            #Decompress all the count_matrices archives and save them just as .tsv files. The new .tsv files are stored
            # in data/HER2_data/count_matrix folder.
            count_matrices_gz_files = glob.glob(os.path.join(SPARED_PATH, 'data','andersson_human_breast_cancer_data', 'unzipped_HER2_count_matrices', '*.gz'))
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'count_matrix'), exist_ok=True)
            for file in count_matrices_gz_files:
                with gzip.open(file, 'rb' ) as f:
                    content = f.read()
                tsv_file_name = file.split('/')[-1][:-3]
                with open(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'count_matrix', tsv_file_name), 'wb') as tsv_file:
                   tsv_file.write(content)
            
            #Delete the HER2_spot_count_matrices.zip file
            os.remove(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_count_matrices.zip"))

            #Delete the unzipper_HER2_spot_matrices folder
            shutil.rmtree(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_count_matrices"))

            #Use wget to download the spot_coordinates
            wget.download('https://zenodo.org/record/4751624/files/spot-selections.zip?download=1', out= os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_spot_coordinates.zip"))
            
            #Unzip the spot_coordinates and save it in other folder
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'unzipped_HER2_spot_coordinates'), exist_ok=True)
            with pyzipper.AESZipFile(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_spot_coordinates.zip"), 'r', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zip_ref:
                try:
                    zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"unzipped_HER2_spot_coordinates"), pwd=str.encode('yUx44SzG6NdB32gY'))
                except: 
                    pass
            
            #Decompress all the spot_coordinates archives and save them just as .tsv files. The new .tsv files are stored
            # in data/HER2_data/spot_coordinates folder.
            spot_coordinates_gz_files = glob.glob(os.path.join(SPARED_PATH, 'data','andersson_human_breast_cancer_data', 'unzipped_HER2_spot_coordinates', '*.gz'))
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'spot_coordinates'), exist_ok=True)
            for file in spot_coordinates_gz_files:
                with gzip.open(file, 'rb' ) as f:
                    content = f.read()
                tsv_file_name = file.split('/')[-1][:-3]
                with open(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'spot_coordinates', tsv_file_name), 'wb') as tsv_file:
                  tsv_file.write(content)
            
            #Delete the HER2_spot_coordinates.zip file
            os.remove(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"HER2_spot_coordinates.zip"))

            #Delete the unzipper_HER2_spot_coordinates folder
            shutil.rmtree(os.path.join(SPARED_PATH, "data", "andersson_human_breast_cancer_data", "unzipped_HER2_spot_coordinates"))


            #Use wget to download the metadata
            wget.download('https://zenodo.org/record/4751624/files/meta.zip?download=1', out= os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data'))

            #Decompress the meta.zip file and save it in data/HER2_data/meta/data folder
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'meta', 'data'), exist_ok = True)
            with pyzipper.AESZipFile(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"meta.zip"), 'r', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zip_ref:
                try:
                    zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data', 'meta', 'data'), pwd=str.encode('yUx44SzG6NdB32gY'))
                except: 
                    pass
            #Delete the meta.zip file
            os.remove(os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data',"meta.zip"))
        return os.path.join(SPARED_PATH, 'data', 'andersson_human_breast_cancer_data')

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


    def get_adata_for_slide(self, patient_id: str) -> ad.AnnData:
        """
        This function loads the data from the given patient_id and replicate and returns an AnnData object all the with relevant information.
        No image patch information is added by this function. It also computes the quality control metrics of the adata object inplace.
        Finally it uses the compute_moran function to compute and add to the var attribute various statistics related to the Moran's I test.
        In case the data is already computed, it is loaded from the processed_data folder.

        Args:
            patient_id (str): The patient id of the patient to load the data from.
            Example: A1
        Returns:
            ad.AnnData: An anndata object with the information of the corresponding slice
        """

        #Based on slide_id get the paths to the files to load
        path_dict = {
            'count_matrix': os.path.join(self.download_path, 'count_matrix', patient_id + '.tsv'),
            'spot_coordinates': os.path.join(self.download_path, 'spot_coordinates', patient_id + '_selection.tsv'),
            'histology_image': os.path.join(self.download_path, 'histology_image', patient_id + '.jpg')
        }

        # We load the count matrix,  spot coordinates and histology image
        count_matrix =  pd.read_table(path_dict['count_matrix'], delimiter='\t')
        spot_coordinates = pd.read_table(path_dict['spot_coordinates'], delimiter='\t')
        histology_image = plt.imread(path_dict['histology_image'])

        
        """#Calculte the value of the Diameter of spot in full resolution
        #Extract two adjacent spots (10x13 and 10x14)
        adjacent_spots = [spot.replace('x', '_').split('_') for spot in count_matrix['Unnamed: 0'].head(2).tolist()]
        #Found the spot in spot_coordinates and extract the pixel space corresponding values
        pixel_space_spots = []
        for spot in adjacent_spots:
            filas_filtradas = spot_coordinates.loc[(spot_coordinates['x'] == int(spot[0])) & (spot_coordinates['y'] == int(spot[1]))]
            pixel_space_spots.append((int(filas_filtradas['pixel_x']), int(filas_filtradas['pixel_y'])))
        diameter_spot = pixel_space_spots[1][1] - pixel_space_spots[0][1]
        print(f'The spot diameter for this dataset is {diameter_spot}')"""


        # Standardize the index of the count_matrix and spot_coordinates (format patient_id_replicate_x_y)
        spot_coordinates.index = [f'{patient_id[0]}_{patient_id[1]}_{spot_coordinates.iloc[i,0]}_{spot_coordinates.iloc[i,1]}' for i in spot_coordinates.index]
        count_matrix.index = [f'{patient_id[0]}_{patient_id[1]}_{count_matrix.iloc[i,0].replace("x", "_")}' for i in count_matrix.index]

        # We compute the intersection between the indexes of the count_matrix and spot_coordinates
        intersection_idx = count_matrix.index.intersection(spot_coordinates.index)

        # Refine count_matrix and spot_coordinates to only contain spots that are in intersection_idx
        count_matrix = count_matrix.loc[intersection_idx]
        spot_coordinates = spot_coordinates.loc[intersection_idx]


        # Declare obs dataframe
        obs_df = pd.DataFrame({
            'patient': patient_id[0],
            'replicate': patient_id[1],
            'array_row': spot_coordinates['y'],
            'array_col': spot_coordinates['x'],
            'tumor': 'none'
        })


        # Set the index name to spot_id
        obs_df.index.name = 'spot_id'

        #Get the var_df from the count_matrix
        del count_matrix['Unnamed: 0']
        var_df = count_matrix.columns.to_frame()
        var_df.index.name = 'gene_ids'
        var_df.columns = ['gene_ids']

        # Declare uns,spatial,sample,metadata dictionary 
        metadata_dict = {
            'chemistry_description': "Spatial Transcriptomics",
            'software_version': 'NA',
            'source_image_path': path_dict['histology_image']
        }

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
        scalefactors_dict = {
            'fiducial_diameter_fullres': 'NA',
            'spot_diameter_fullres': 222.0, # This diameter was adjusted by looking at the scatter plot of the spot coordinates
            'tissue_hires_scalef': 2000/histology_image.size[0],
            'tissue_lowres_scalef': 600/histology_image.size[0]
        }

        uns_dict = {
            'spatial': {
                patient_id: {
                    'metadata': metadata_dict,
                    'scalefactors': scalefactors_dict,
                    'images': images_dict
                }

            },
            'cancer_type': "None"
        }

        #Set the spot_coordinates just with pixel_x and pixel_y information
        spot_coordinates = spot_coordinates[['pixel_x',	'pixel_y']].astype(int)

        #Create obsm_dict with the corresponding values
        obsm_dict = {
            'spatial': spot_coordinates.values
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
        # Get the spatial coordinates of the centers of the spots and set the index
        coord =  pd.DataFrame(adata.obsm['spatial'], columns=['x_coord', 'y_coord'], index = adata.obs_names)
        
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
        flat_patches = np.zeros((adata.n_obs, window**2*3), dtype=np.uint8)

        # Iterate over the coordinates and get the patches
        for i, (x, y) in enumerate(coord.values):
            # Get the patch
            x = int(x)
            y = int(y)
            patch = hires_img[y - (window//2):y + (window//2), x - (window//2):x + (window//2), :]
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
        print("The first time running this function will take around 4 minutes to read the data in adata format.")
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
                    # Finally, just leave the patches of the scale that is being used. Remove the rest
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

        #Read the archive in annotations/mapping_HUGO_2_Ensemble.txt, this contains the information to pass from
        # HUGO symbol to Ensemble gene notation
        df_names = pd.read_csv(os.path.join(SPARED_PATH, 'data', 'annotations', 'mapping_HUGO_2_Ensemble.txt'), delimiter='\t')
        #Create a dictionary with key = HUGO symbol and value the respective Ensemble gene notation
        dict_HUGO2Ensemble = df_names.set_index('Approved symbol')['Ensembl gene ID'].to_dict()
        df_names = df_names.set_index('Approved symbol')

        #Calculate the intersection between the indexes HUGO symbols in the file and the annData object
        intersection_idx = slide_collection.var_names.intersection(df_names.index)

        #Obtain the HUGO that have valid Ensemble equivalent, no nan values are allowed
        filtered_index = [element for element in intersection_idx.tolist() if isinstance(dict_HUGO2Ensemble[element], str)]

        #Obtain the genes that are posible to use thanks to they have a valid Ensemble equivalent
        new_slide_collection = slide_collection[:, filtered_index].copy()
        
        #Change the var_names from HUGO symbols to Ensemble genes notation
        new_slide_collection.var_names = pd.Index([ dict_HUGO2Ensemble[HUGO] for HUGO in new_slide_collection.var_names])

        # Return the patient collection
        return new_slide_collection