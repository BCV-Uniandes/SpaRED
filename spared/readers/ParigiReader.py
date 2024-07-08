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
import shutil
import scipy
import tifffile
import tarfile
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

class ParigiReader():
    def __init__(self,
        dataset: str = 'parigi_mouse_intestine',
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
        Parigi et al. (https://doi.org/10.1038/s41467-022-28497-0) which is hosted in https://doi.org/10.1038/s41467-022-28497-0

        Args:
            dataset (str, optional): An string encoding the dataset type. In this case only 'parigi_mouse_intestine' will work. 
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
        self.hex_geometry = False if self.dataset == 'parigi_mouse_intestine' else True

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
        
        # Get names dictionary
        names_dict = {
            'train': ["B1_S2"],
            'val':   ["A1_S1"],
            'test':  [] 
        }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict 

    def download_data(self) -> str:
        """
        This is a reader class that can download data, get adata objects and compute a collection of slides into an adata object. It is limited to
        reading and will not perform any processing or filtering on the data. In this particular case, it will read the data from the dataset published by
        Parigi et al. (https://doi.org/10.1038/s41467-022-28497-0) which is hosted in https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169749

        Returns:
            str: Path to the data directory.
        """
        # Use wget to download the data
        data_path = os.path.join(SPARED_PATH, 'processed_data', 'parigi_data')
        if not os.path.exists(data_path) or self.force_compute:
            os.makedirs(data_path, exist_ok=True)
            tar_path = os.path.join(SPARED_PATH, 'data', 'parigi_data','GSE169749_RAW.tar')
            wget.download('https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE169749&format=file', out=tar_path)                      
            
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=data_path)

            # Delete the tar file
            os.remove(tar_path)

            gz_files = [file for file in os.listdir(data_path) if file.endswith('.gz')]
            for gz_file in gz_files:
                gz_path = os.path.join(data_path, gz_file)
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(gz_path[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gz_path) 

            # Divide slides information
            slide_train = self.split_names['train'][0]
            slide_val = self.split_names['val'][0]

            if not os.path.exists(os.path.join(data_path, slide_train)):
                os.makedirs(os.path.join(data_path, slide_train), exist_ok=True)

            if not os.path.exists(os.path.join(data_path, slide_val)):
                os.makedirs(os.path.join(data_path, slide_val), exist_ok=True)

            all_files = [file for file in os.listdir(data_path)]
            for file in all_files:
                if slide_train in file:
                    shutil.move(os.path.join(data_path, file), os.path.join(data_path, slide_train))
                elif slide_val in file:
                    shutil.move(os.path.join(data_path, file), os.path.join(data_path, slide_val))
           
        return os.path.join(SPARED_PATH, 'data', 'parigi_data') 

    def get_or_save_dataset_path(self) -> str:
        """
        This function saves the parameters of the dataset in a dictionary on a path in the
        processed_dataset folder. The path is returned.

        Returns:
            str: Path to the saved parameters.
        """

        # Get all the class attributes of the current dataset
        curr_dict = self.__dict__.copy()
        # Delete metadata and count matrix from dictionary
        curr_dict.pop('metadata', None)
        curr_dict.pop('count_matrix', None)
        
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
        This function loads the data from the given patient_id and replicate and returns an AnnData object all the with relevant information.
        No image patch information is added by this function. It also computes the quality control metrics of the adata object inplace.
        Finally it uses the compute_moran function to compute and add to the var attribute various statistics related to the Moran's I test.
        In case the data is already computed, it is loaded from the processed_data folder.

        Args:
            slide_id (str): Slide id to load the data from
        """ 

        if slide_id == "A1_S1":
            sample = "GSM5213483"
            region = "Region1_colon_d0"
        else:
            sample = "GSM5213484"
            region = "Region2_colon_d14"

        # Get the paths to the files to load
        path_dict = {
            'spot_coordinates': os.path.join(self.download_path, slide_id, f'{sample}_V19S23-097_{slide_id}_tissue_positions_list.csv'),
            'histology_image': os.path.join(self.download_path, slide_id, f'{sample}_V19S23-097_{slide_id}_{region}_rotated.tif'),
            'count_matrix': os.path.join(self.download_path, slide_id, f'{sample}_V19S23-097_{slide_id}_normalized_data.tsv'),
            'scale_factors': os.path.join(self.download_path, slide_id, f'{sample}_V19S23-097_{slide_id}_scalefactors_json.json')
        }

        # We load the histology image
        histology_image = tifffile.imread(path_dict['histology_image'])

        # We load ensg names for translation txt
        ensg_names = pd.read_csv(os.path.join(SPARED_PATH, 'data', 'annotations', 'mart_export.txt'), sep="\t")

        # We load the count matrix
        count_matrix = pd.read_csv(path_dict['count_matrix'], sep="\t")
        count_matrix = count_matrix.set_index(count_matrix.columns[0])

        # We make the match
        map_df = count_matrix.index.to_frame()
        map_df.columns = ['gene_ids']
        map_df = map_df[map_df['gene_ids'].isin(ensg_names['Gene name'])]
        map_df = map_df.merge(ensg_names, left_index=True, right_on='Gene name', how='left')
        map_df = map_df.set_index('gene_ids')
        map_df = map_df.drop("Gene name", axis=1)
        map_df.columns = ['gene_ids']

        # Redefine count matrix
        count_matrix = count_matrix[count_matrix.index.isin(map_df.index)]
        count_matrix = np.transpose(count_matrix)
        count_matrix.index = count_matrix.index.str.split('_').str[0]

        # We load the spot coordinates 
        names = ['gene_ids','c1','c2','c3','pixel_y','pixel_x']
        spot_coordinates = pd.read_csv(path_dict['spot_coordinates'], sep=",", names=names)
        spot_coordinates = spot_coordinates[spot_coordinates["gene_ids"].isin(count_matrix.index)]
        spot_coordinates = spot_coordinates.reset_index()
        indexes = spot_coordinates.index.values
        indexes = [slide_id+"_"+str(i) for i in indexes]
        spot_coordinates.index = indexes

        # Declare obs dataframe
        obs_df = pd.DataFrame({
            'patient': slide_id,
            'array_row': spot_coordinates['pixel_y'],
            'array_col': spot_coordinates['pixel_x'],
        })
        # Set the index name to spot_id
        obs_df.index.name = 'spot_id'

        #### Declare uns,spatial,sample,metadata dictionary 
        metadata_dict = {
            'chemistry_description': "Spatial Transcriptomics",
            'software_version': 'NA',
            'source_image_path': path_dict['histology_image']
        }

        # We load the scale factors
        with open(path_dict['scale_factors']) as json_file:
            scalefactors_dict = json.load(json_file)

        ## Reshape histology image to lowres and hires
        # Read image into PIL
        histology_image = histology_image.astype(np.float32) / 65535.0
        histology_image = (histology_image * 255).astype(np.uint8)
        histology_image = histology_image.astype(np.uint8)
        histology_image = Image.fromarray(histology_image)
        # Resize to lowres
        lowres_factor = scalefactors_dict["tissue_lowres_scalef"]
        histology_image_lowres = histology_image.resize((int(histology_image.width * lowres_factor), int(histology_image.height * lowres_factor)))
        # Resize to hires
        hires_factor = scalefactors_dict["tissue_hires_scalef"]
        histology_image_hires = histology_image.resize((int(histology_image.width * hires_factor), int(histology_image.height * hires_factor)))
        # Convert to array
        histology_image_lowres = np.array(histology_image_lowres)
        histology_image_hires = np.array(histology_image_hires)
        # Create images dictionary
        images_dict = {
            'hires': histology_image_hires,
            'lowres': histology_image_lowres
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

        obsm_dict = {'spatial': spot_coordinates[["pixel_x", "pixel_y"]].values}

        # Drop duplicates
        duplicated_mask = map_df.index.duplicated(keep=False)
        duplicated_rows = map_df[duplicated_mask]
        duplicated_genes = np.unique(duplicated_rows.index.values)
        var_df = map_df[~duplicated_mask]

        # Declare a scipy sparse matrix from the count matrix without the duplicates
        count_matrix.drop(duplicated_genes, axis=1, inplace=True)
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
        print("The first time running this function will take some time to read the data in adata format.")
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
                    # Add the key patient to a column in the obs
                    adata.obs['patient'] = slide_id.split('_')[1]
                    # Add a unique ID column to the observations to be able to track them when in cuda
                    adata.obs['unique_id'] = adata.obs.index
                    # Change the var_names to the gene_ids
                    adata.var_names = adata.var['gene_ids']
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

        # Return the patient collection
        return slide_collection