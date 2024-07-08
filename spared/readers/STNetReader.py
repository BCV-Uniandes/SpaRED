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
class STNetReader():
    def __init__(self,
        dataset: str = 'stnet_dataset',
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
        reading and will not perform any processing or filtering on the data. In this particular case, it will read the data from the STNet dataset.

        Args:
            dataset (str, optional): An string encoding the dataset type. In this case only 'stnet_dataset' will work. Defaults to 'stnet_dataset'.
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
        if the data split. 
        Returns:
            dict: Dictionary of data names for train, validation and test in lists.
        """
        
        # Get names dictionary
        names_dict = {
            'train': ["BC23287_C1","BC23287_C2","BC23287_D1","BC23450_D2","BC23450_E1",
                      "BC23450_E2","BC23944_D2","BC23944_E1","BC23944_E2","BC24220_D2",
                      "BC24220_E1","BC24220_E2","BC23567_D2","BC23567_E1","BC23567_E2",
                      "BC23810_D2","BC23810_E1","BC23810_E2","BC23903_C1","BC23903_C2",
                      "BC23903_D1","BC24044_D2","BC24044_E1","BC24044_E2","BC24105_C1",
                      "BC24105_C2","BC24105_D1","BC23269_C1","BC23269_C2","BC23269_D1",
                      "BC23272_D2","BC23272_E1","BC23272_E2","BC23277_D2","BC23277_E1",
                      "BC23277_E2","BC23895_C1","BC23895_C2","BC23895_D1","BC23377_C1",
                      "BC23377_C2","BC23377_D1","BC23803_D2","BC23803_E1","BC23803_E2"],
            'val':   ["BC23901_C2","BC23901_D1","BC24223_D2","BC24223_E1","BC24223_E2",
                      "BC23270_D2","BC23270_E1","BC23270_E2","BC23209_C1","BC23209_C2",
                      "BC23209_D1"],
            'test':  ["BC23268_C1","BC23268_C2","BC23268_D1","BC23506_C1","BC23506_C2",
                      "BC23506_D1","BC23508_D2","BC23508_E1","BC23508_E2","BC23288_D2",
                      "BC23288_E1","BC23288_E2"] 
        }

        # Print the names of the datasets
        print(f'Loading {self.dataset} dataset with the following data split:')
        for key, value in names_dict.items():
            print(f'{key} data: {value}')

        return names_dict 

    def download_data(self) -> str:
        """
        This function downloads the data of the original STNet from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/29ntw7sh4r-5.zip
        using wget to the data/STNet_data directory. Then it unzips the file and deletes the zip file. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory.
        """
        # Use wget to download the data
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', 'STNet_data')) or self.force_compute:
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'STNet_data'), exist_ok=True)
            wget.download('https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/29ntw7sh4r-5.zip', out=os.path.join(SPARED_PATH, 'data', 'STNet_data',"29ntw7sh4r-5.zip"))
        
            # Unzip the file in a folder with an understandable name
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'STNet_data', 'unzipped_STNet_data'))
            with zipfile.ZipFile(os.path.join(SPARED_PATH, "data", "STNet_data", "29ntw7sh4r-5.zip"), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(SPARED_PATH, 'data', 'STNet_data', 'unzipped_STNet_data'))
            
            # Delete the zip file
            os.remove(os.path.join(SPARED_PATH, "data", "STNet_data", "29ntw7sh4r-5.zip"))

            # There is an extra folder inside the unzipped folder. We move the files to the unzipped folder.
            files = os.listdir(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", "Human breast cancer in situ capturing transcriptomics"))
            for file in files:
                shutil.move(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", "Human breast cancer in situ capturing transcriptomics",file),os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data"))

            # We delete the extra folder           
            shutil.rmtree(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", "Human breast cancer in situ capturing transcriptomics"))

            # Create folders in STNet_data for count_matrix, histology_image, spot_coordinates and tumor_annotation
            folder_names = ['count_matrix', 'histology_image', 'spot_coordinates', 'tumor_annotation']
            [os.makedirs(os.path.join(SPARED_PATH, 'data', 'STNet_data', f), exist_ok=True) for f in folder_names]

            # move the metadata csv to STNet_data
            shutil.move(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", "metadata.csv"), os.path.join(SPARED_PATH, "data", "STNet_data")) 

            # Read the metadata csv
            metadata = pd.read_csv(os.path.join(SPARED_PATH, 'data', 'STNet_data', 'metadata.csv'))

            # Iterate over the folder names to unzip the files in the corresponding folder
            for f in folder_names:
                # get filenames from the metadata column
                file_names = metadata[f]
                # If f is histology_image move the files to the histology_image folder
                if f == 'histology_image':
                    [shutil.move(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", fn),os.path.join(SPARED_PATH, "data", "STNet_data", f, fn)) for fn in file_names]
                # If any other folder, extract the .gz files
                else:
                    for fn in file_names:
                        with gzip.open(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", fn), 'rb') as f_in:
                            with open(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data",fn[:-3]), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    # move the files to the corresponding folder
                    [shutil.move(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data", fn[:-3]),os.path.join(SPARED_PATH, "data", "STNet_data", f, fn[:-3])) for fn in file_names]

            # We delete the unzipped folder
            shutil.rmtree(os.path.join(SPARED_PATH, "data", "STNet_data", "unzipped_STNet_data"))
            
        return os.path.join(SPARED_PATH, 'data', 'STNet_data') 

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
        This function loads the data from the given patient_id and replicate and returns an AnnData object all the with relevant information.
        No image patch information is added by this function. It also computes the quality control metrics of the adata object inplace.
        Finally it uses the compute_moran function to compute and add to the var attribute various statistics related to the Moran's I test.
        In case the data is already computed, it is loaded from the processed_data folder.

        Args:
            patient_id (str): The patient id of the patient to load the data from.
            replicate (str): The replicate to load the data from.
        """
        # Get the patient and replicate from the slide name
        patient_id, replicate = slide_id.split('_')

        # Read the metadata csv
        metadata = pd.read_csv(os.path.join(self.download_path, 'metadata.csv'))

        # Get the row of the patient_id and replicate in the metadata
        slide_row = metadata[(metadata['patient'] == patient_id) & (metadata['replicate'] == replicate)]
        # Get the paths to the files to load
        path_dict = {
            'count_matrix': os.path.join(self.download_path, 'count_matrix', slide_row.count_matrix.item()[:-3]),
            'tumor_annotation': os.path.join(self.download_path, 'tumor_annotation', slide_row.tumor_annotation.item()[:-3]),
            'spot_coordinates': os.path.join(self.download_path, 'spot_coordinates', slide_row.spot_coordinates.item()[:-3]),
            'histology_image': os.path.join(self.download_path, 'histology_image', slide_row.histology_image.item())
        }
        
        # We load the count matrix, tumor annotation, spot coordinates and histology image
        count_matrix = pd.read_csv(path_dict['count_matrix'], index_col = 0, sep='\t', header=0, engine="pyarrow")
        tumor_annotation = pd.read_csv(path_dict['tumor_annotation'], index_col = 0, sep='\t', header=0)
        spot_coordinates = pd.read_csv(path_dict['spot_coordinates'], index_col = 0)
        histology_image = plt.imread(path_dict['histology_image'])

        # Correct tumor_annotation columns by shifting them one to the right
        tumor_annotation.columns = [tumor_annotation.index.name] + tumor_annotation.columns[:-1].to_list()
        tumor_annotation.index.name = None

        # Round the 'xcoord' and 'ycoord' columns of the tumor_annotation to integers
        tumor_annotation['xcoord'] = tumor_annotation['xcoord'].round().astype(int)
        tumor_annotation['ycoord'] = tumor_annotation['ycoord'].round().astype(int)

        # Update tumor_annotation index with the rounded xcoord and ycoord
        tumor_annotation.index = [f'{i.split("_")[0]}_{tumor_annotation.loc[i, "xcoord"]}_{tumor_annotation.loc[i, "ycoord"]}' for i in tumor_annotation.index]

        # Standardize the index of the count_matrix, tumor_annotation and spot_coordinates (format patient_id_replicate_x_y)
        tumor_annotation.index = [f'{patient_id}_{i}' for i in tumor_annotation.index]
        count_matrix.index = [f'{patient_id}_{replicate}_{i.replace("x", "_")}' for i in count_matrix.index]  
        spot_coordinates.index = [f'{patient_id}_{replicate}_{i.replace("x", "_")}' for i in spot_coordinates.index]

        # We compute the intersection between the indexes of the count_matrix, tumor_annotation and spot_coordinates
        intersection_idx = count_matrix.index.intersection(spot_coordinates.index).intersection(tumor_annotation.index)

        # Refine tumor_annotation, count_matrix and spot_coordinates to only contain spots that are in intersection_idx
        tumor_annotation = tumor_annotation.loc[intersection_idx]
        count_matrix = count_matrix.loc[intersection_idx]
        spot_coordinates = spot_coordinates.loc[intersection_idx]
 

        #### Declare obs dataframe
        obs_df = pd.DataFrame({
            'patient': patient_id,
            'replicate': replicate,
            'array_row': tumor_annotation['ycoord'],
            'array_col': tumor_annotation['xcoord'],
            'tumor': tumor_annotation['tumor']=='tumor'
        })
        # Set the index name to spot_id
        obs_df.index.name = 'spot_id'

        #### Get the var_df from the count_matrix
        var_df = count_matrix.columns.to_frame()
        var_df.index.name = 'gene_ids'
        var_df.columns = ['gene_ids']

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
        scalefactors_dict = {
            'fiducial_diameter_fullres': 'NA',
            'spot_diameter_fullres': 300.0, # This diameter was adjusted by looking at the scatter plot of the spot coordinates
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

            },
            'cancer_type': slide_row.type.item()
        }

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
        print("The first time running this function will take around 10 minutes to read the data in adata format.")
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

        # Return the patient collection
        return slide_collection