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
import matplotlib.pyplot as plt
import pathlib

# Remove the max limit of pixels in a figure
Image.MAX_IMAGE_PIXELS = None

# Get the path of the spared database
SPARED_PATH = pathlib.Path(__file__).parents[1]

# NOTE: This class is not being currently used nor tested because it is from the ST technology.
class MoncadaReader():
    def __init__(self,
        dataset: str = "moncada_human_PDAC_dataset",
        #REVISAR
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
        force_compute: bool = False,
        ):

        """
        This is a reader class that can download the data from HER2+ dataset.
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
        """names_dict = {
            'train': ['GSM3036911', 'GSM3405534', 'GSM4100721',
                      'GSM4100722', 'GSM4100723', 'GSM4100724'],
            'val':   ['GSM4100725', 'GSM4100726'],
            'test':  ['GSM4100727', 'GSM4100728'] 
        }"""

        names_dict = {
            'train': ['GSM4100725', 'GSM4100726'],
            'val':   ['GSM4100727'],
            'test':  ['GSM4100728'] 
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
        This function downloads all the ST data related with this paper https://doi.org/10.1038/s41587-019-0392-8.
        using wget to the data/moncada_human_PDAC_data directory. Then it unzips the files and deletes the zip files. This function returns a string with the path where the data is stored.

        Returns:
            str: Path to the data directory with the images and count_matrix
        """
        if not os.path.exists(os.path.join(SPARED_PATH, 'processed_data', 'moncada_human_PDAC_data')) or self.force_compute:
            print("The download process of the PDAC-ST dataset takes approximately 3 - 4 min.")
            #First we create a dictionary with the information to access the download links
            #This dictionary has key: patient name, values are a list with this information [type (A or B), replicate]
            supplementary_dict = { 'GSM3036911': ['A', '1'], 'GSM3405534': ['B', '1'], 'GSM4100721': ['A', '2'],
                                   'GSM4100722': ['A', '3'], 'GSM4100723': ['B', '2'], 'GSM4100724': ['B', '3'],
                                   'GSM4100725': ['D', '', '1'], 'GSM4100726': ['E', '', '1'], 'GSM4100727': ['F', '', '1'],
                                   'GSM4100728': ['G', '', '1']}
            
            #Create the folder to save all the data
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data'), exist_ok=True)

            #Create the folder to save the compressed images
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'compressed_images'), exist_ok=True)

            #Creatae the folder to save the compressed count_matrix
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'compressed_count_matrix'), exist_ok=True)
            
            #Use wget to download the images and count_matrix files using the information in the dictionary
            for patient in list(supplementary_dict.keys()):
                patient_type = supplementary_dict[patient][0]
                replicate = supplementary_dict[patient][1]
                try:
                    aux_cm = supplementary_dict[patient][2]
                except:
                    aux_cm = replicate
                
                #Download the images
                link_img = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={patient}&format=file&file={patient}%5FPDAC%2D{patient_type}%2DST{replicate}%2DHE%2Ejpg%2Egz'
                wget.download(link_img, out=os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'compressed_images'))

                #Download the count_matrix
                if patient_type == 'A' and replicate == '1':
                    link_cm = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={patient}&format=file&file={patient}%2Etsv%2Egz'
                elif patient_type == 'B' and replicate == '1' :
                    link_cm = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={patient}&format=file&file={patient}%5FPDAC%2D{patient_type}%2DST{replicate}%2Etsv%2Egz'
                else:
                    link_cm = f'https://www.ncbi.nlm.nih.gov/geo/download/?acc={patient}&format=file&file={patient}%5FPDAC%2D{patient_type}%2Dst{aux_cm}%2Etsv%2Egz'
                wget.download(link_cm, out=os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'compressed_count_matrix'))

            #Create new folder to save the decompressed images
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'histology_image'), exist_ok=True)

            #We decompress the images and save them in the desired folder
            images_gz_files = glob.glob(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'compressed_images', '*.gz'))
            for img_gz in tqdm(images_gz_files):
                img_name = img_gz.split('/')[-1][:-3]
                with gzip.open(img_gz, 'rb') as f_in:
                    with open(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'histology_image', img_name), 'wb') as f_out:
                        f_out.write(f_in.read())
            
            #Delete the folder that have the compressed images
            shutil.rmtree(os.path.join(SPARED_PATH, "data", 'moncada_human_PDAC_data', "compressed_images"))

            #Create the folder to save the decompressed count_matrix
            os.makedirs(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'count_matrix'), exist_ok=True)
                
            #We decompress the count_matrix and save them in the desired folder
            count_matrices_gz_files = glob.glob(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'compressed_count_matrix', '*.gz'))
            for file in tqdm(count_matrices_gz_files):
                with gzip.open(file, 'rb' ) as f:
                    content = f.read()
                tsv_file_name = file.split('/')[-1][:-3]
                with open(os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data', 'count_matrix', tsv_file_name), 'wb') as tsv_file:
                   tsv_file.write(content)

            #Delete the folder that have the compressed count_matrix
            shutil.rmtree(os.path.join(SPARED_PATH, "data", 'moncada_human_PDAC_data', 'compressed_count_matrix'))

            #we process the file names to simplify them without losing the ability to differentiate between them
            # Change the names of the count_matrix files

            count_matrix_files_original = glob.glob(os.path.join(SPARED_PATH, "data", "moncada_human_PDAC_data" , "count_matrix", "*.tsv"))
            for cm_name in count_matrix_files_original:
                new_name = cm_name.split(os.sep)[-1].split(".")[0].split("_")[0]
                aux_list = cm_name.split(os.sep)[:-1]
                aux_list.append(new_name + ".tsv")
                new_path = os.sep.join(aux_list)
                os.rename(cm_name, new_path)

            #Change the names of the histology image files
            images_files_original = glob.glob(os.path.join(SPARED_PATH, "data", "moncada_human_PDAC_data" , "histology_image", "*.jpg"))
            for img_name in images_files_original:
                new_name = img_name.split(os.sep)[-1].split(".")[0].split("_")[0]
                aux_list = img_name.split(os.sep)[:-1]
                aux_list.append(new_name + ".jpg")
                new_path = os.sep.join(aux_list)
                os.rename(img_name, new_path)
        
        return os.path.join(SPARED_PATH, 'data', 'moncada_human_PDAC_data')

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
        This function loads the data from the given patient_id and returns an AnnData object all the with relevant information.
        No image patch information is added by this function.
        In case the data is already computed, it is loaded from the processed_data folder.

        Args:
            patient_id (str): The patient id of the patient to load the data from.
            Example: GSM3036911
        Returns:
            ad.AnnData: An anndata object with the information of the corresponding slice
        """
        #Based on slide_id get the paths to the files to load
        path_dict = {
            'count_matrix': os.path.join(self.download_path, 'count_matrix', patient_id + '.tsv'),
            'histology_image': os.path.join(self.download_path, 'histology_image', patient_id + '.jpg'),
            'spot_coordinate': os.path.join(self.download_path, 'spot_coordinates', patient_id + '.txt')
        }

        #We load the files: count_matrix and image
        count_matrix =  pd.read_table(path_dict['count_matrix'], delimiter='\t')
        histology_image = plt.imread(path_dict['histology_image'])
        if "x" in count_matrix.columns.to_list()[-1]:
            count_matrix =  pd.read_table(path_dict['count_matrix'], delimiter='\t', index_col="Genes")
            count_matrix = count_matrix.T
            count_matrix.index = [f'{patient_id}_{patient_id[-2:]}_{i.replace("x", "_")}' for i in count_matrix.index]
            diff = True
        else:
            count_matrix.index = [f'{patient_id}_{patient_id[-2:]}_{count_matrix.iloc[i,0].replace("x", "_")}' for i in count_matrix.index]
            diff = False

        """#We create the spot_coordinates using the linear transformation
        x = np.array([])
        y = np.array([])
        for index in count_matrix.index.to_list():
            partes = index.split("_")
            try:
                y = np.append(y, int(partes[3]))
                x = np.append(x, int(partes[2]))
            except:
                pass
        constants = {"GSM4100725":  {'m_x': 350, 'm_y': 350, 'a_x': 2000, 'a_y':  3000, "spot_diameter_fullres":230 },
                     "GSM4100723":  {'m_x': 350, 'm_y': 350, 'a_x': 700, 'a_y':  700, "spot_diameter_fullres":230},
                     "GSM4100721":  {'m_x': 650, 'm_y': 650, 'a_x': 600, 'a_y':  1200, "spot_diameter_fullres":400},
                     "GSM4100722":  {'m_x': 800, 'm_y': 800, 'a_x': 1700, 'a_y':  1500, "spot_diameter_fullres":400}}
        try:
            print("---------------------------------------------------------------------------------")
            values = constants[patient_id]
            pixel_X = values["m_x"] * x + values["a_x"]
            pixel_Y = values["m_y"] * y + values["a_y"]
        except:
            pixel_X = 294*x - 294
            pixel_Y = 294*y - 294"""
        
        constants = {"GSM4100725": 250,
                     "GSM4100726": 250,
                     "GSM4100727": 250,
                     "GSM4100728": 250}

        file_pixel_space = pd.read_table(path_dict['spot_coordinate'], delimiter='\t')
        file_pixel_space = file_pixel_space.drop(" ", axis=1)
        spot_coordinates = file_pixel_space.set_index(count_matrix.index)
        new_names = ["pixel_x",	"pixel_y"]
        spot_coordinates.columns = new_names

        x = []
        y = []
        for i in spot_coordinates.index.to_list():
            x.append(int(float(i.split("_")[-2])))
            y.append(int(float(i.split("_")[-1])))

        #Create the spot_coordinates dataFrame, tenga en cuenta que las coordenadas se giran.
        #spot_coordinates = pd.DataFrame({'x': y, 'y': x, 'pixel_x': pixel_Y, 'pixel_y': pixel_X})
        spot_coordinates = spot_coordinates.astype(int)
        spot_coordinates['x'] = x
        spot_coordinates['y'] = y
    
        # We compute the intersection between the indexes of the count_matrix and spot_coordinates
        intersection_idx = count_matrix.index.intersection(spot_coordinates.index)

        # Refine count_matrix and spot_coordinates to only contain spots that are in intersection_idx
        count_matrix = count_matrix.loc[intersection_idx]
        spot_coordinates = spot_coordinates.loc[intersection_idx]
        
        #Delete duplicated columns if necessary
        count_matrix = count_matrix.loc[:, ~count_matrix.columns.duplicated()]

        # Declare obs dataframe
        obs_df = pd.DataFrame({
            'patient': patient_id,
            'replicate': patient_id[-2:],
            'array_row': spot_coordinates['y'],
            'array_col': spot_coordinates['x'],
            'tumor': 'none'
        })



        # Set the index name to spot_id
        obs_df.index.name = 'spot_id'

        #Get the var_df from the count_matrix
        if diff:
            pass
        else:
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

        #Reshape histology image to lowres (600, 600, 3) and hires (2000, 2000, 3)
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
            'spot_diameter_fullres': constants[patient_id], # This diameter was adjusted by looking at the scatter plot of the spot coordinates
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