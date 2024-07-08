#File created for automate generation of all dataset pages
#The base is the Abalo dataset page

import numpy as np
import pandas as pd
import os
import glob
import shutil
from bs4 import BeautifulSoup

def load_csv_file(csv_path:str) -> pd.DataFrame:
    """
    Function to load .csv file with all the information about each dataset into a dataframe.
    Preprocess the dataframe to have better order.
    Args:
        -csv_path (str): path to the csv file.
    Returns:
        -dataframe (pd.DataFrame): dataframe with all the information about the datasets
    """
    #Load the csv file
    dataframe = pd.read_csv(csv_path)
    #Load the names of the methods
    methods_names = dataframe.loc[0].iloc[12:22].tolist()
    #Change the names of the columns for MSE values
    dataframe.rename(columns={'MSE': 'ShuffleNet_MSE'}, inplace=True)

    methods_index = 1
    for i in range(13, 22):
        dataframe.rename(columns={f'Unnamed: {i}': f'{methods_names[methods_index]}_MSE'}, inplace=True)
        methods_index += 1
    #Change the names of the columns for PCC values
    dataframe.rename(columns={'PCC': 'ShuffleNet_PCC'}, inplace=True)
    methods_index = 1
    for i in range(23, 32):
        dataframe.rename(columns={f'Unnamed: {i}': f'{ methods_names[methods_index]}_PCC'}, inplace=True)
        methods_index += 1
    
    #Remove the first row of the dataset
    dataframe.drop(0, inplace=True)
    
    return dataframe


def create_folders(dataframe: pd.DataFrame, copy: bool) -> None:
    """
    This function creates folders for each dataset if its neccesary. Also copies the images from Abalo dataset, this is only for practical purposes,
    its expected to replace this images with plots.
    Args:
        -dataframe (pd.DataFrame): dataframe with all the information of the datasets
    """
    #Extract all the abbreviations (folder names)
    abbreviations = dataframe['Abbreviation'].tolist()

    #Create the folder for each dataset
    for abb in abbreviations:
        if not os.path.exists(os.path.join('Datasets_Pages', abb)):
            os.makedirs(os.path.join('Datasets_Pages', abb))
        if not os.path.exists(os.path.join('Datasets_Pages', abb, 'Images')):
             os.makedirs(os.path.join('Datasets_Pages', abb, 'Images'))

        #Copy the images of Abalo dataset in other dataset, only once and for test purposes
        if copy and abb != 'AHSCC':
            #Extract all the images of Abalo dataset
            images_paths = glob.glob(os.path.join('Datasets_Pages', 'AHSCC', 'Images', '*.png'))
            for pt in images_paths:
                shutil.copy(pt, pt.replace('AHSCC', abb))


def load_HTML(path: str) -> str:
    """
    This function loads an HTML file as a string. Supuse to loads just the Abalo HTML, since this one its the base
    Args:
        -path (str): path of teh HTML file
    Retuns:
        -text_html (str): text related to the HTML file
    """
    with open(path, "r") as file:
        text_html = file.read()
    return text_html

def leaderboard_dataframe(series: pd.Series) -> pd.DataFrame:
    """
    This function recives a pd.Series with all the information about one dataset and returns pd.DataFrame (columns=Model MSE PCC) (rows= metric values)
    Args:
        -series (pd.Series): series with all the information about a Dataset
    Retuns:
        -df (pd.Dataframe): dataframe wiht the information for the leaderboard
    """

    #Obtain all the MSE metrics
    mse_series = series[series.index.str.contains('MSE')]
    #Obtain all the PCC metrics
    pcc_series = series[series.index.str.contains('PCC')]
    #Obtain models names
    models =  series[series.index.str.contains('PCC')].index.str.replace('_PCC', '').tolist()
    
    #Create new dataframe with the information
    data = {'Model': models,'MSE': mse_series.values.tolist(), 'PCC': pcc_series.values.tolist()}
    df = pd.DataFrame(data)
    for column in ['MSE', 'PCC']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    #Organize the new dataframe by MSE
    df['MSE'] = df['MSE'].round(3)
    df['PCC'] = df['PCC'].round(3)
    df= df.sort_values(by='MSE', ascending=True, na_position='last')
    return df

def dataframe_2_hmtl_table(df: pd.DataFrame)-> str:
    """
    This function creates the tbody of an html table (Leaderboard) based on the information in the dataframe.
    """
    # Reeplace NaN with '-'
    df.fillna('-', inplace=True)
    #Create HTML
    html_table = df.to_html(index=False)
    #Extract tbody content
    tbody = html_table.split('<tbody>')[1].split('</tbody>')[0]

    return tbody


def edit_HTML(text_html: str, dataframe: pd.DataFrame) -> str:
    """
    This function charges the base HTML and based in the dataframe information edits it and create a new dataset page.
    """
    
    #Iterate over the dataframe rows, each row is a dataset
    for index, row in dataframe.iterrows():
        
        if row['Abbreviation'] == 'AHSCC':
            pass
        else:
            base_html_soup  = BeautifulSoup(text_html, "html.parser")

            #Set a new page title: Dataset Name (abbreviation)
            title_id = base_html_soup.find("h1", string='Abalo Human Squamous Cell Carcinoma (AHSCC)')
            title_id.string = f'{row["Name"]} ({row["Abbreviation"]})'

            #New DOI
            doi_id = base_html_soup.find("a", string='https://doi.org/10.17632/2bh5fchcv6.1')
            doi_id.string = row['Doi']
            doi_id['href'] = row['Doi']
            
            #New Access Links
            if row['Link2'] == '-':
                link_id = base_html_soup.find("a", string='https://data.mendeley.com/datasets/2bh5fchcv6/1')
                link_id.string = row['Link1']
                link_id['href'] = row['Link1']
            else:
                link_id = base_html_soup.find("a", string='https://data.mendeley.com/datasets/2bh5fchcv6/1')
                href1 = row['Link1']
                href2 = row['Link2']

                link1 = base_html_soup.new_tag("a", href=href1)
                link1.string = href1
                link1['href'] = href1

                link2 = base_html_soup.new_tag("a", href=href2)
                link2.string = href2
                link2['href'] = href2

                link_id.replace_with(link1)
                link1.insert_after(link2)
            
            #New Abstract
            abstract_id = base_html_soup.find("b", string='Original abstract')
            if abstract_id and abstract_id.parent.name == "p":
                abstract_id = abstract_id.parent
                abstract_id.span.string = row['Abstract']

            #New patients, slides and spots
            #Slides
            slides_id = base_html_soup.find("b", string='Slides')
            if slides_id and slides_id.parent.name == "p":
                slides_id = slides_id.parent
                slides_id.span.string = str(int(row['Slides']))
            #Patients
            patients_id = base_html_soup.find("b", string='Patients')
            if patients_id and patients_id.parent.name == "p":
                patients_id = patients_id.parent
                patients_id.span.string = str(int(row['Patients']))

            #Spots
            spots_id = base_html_soup.find("b", string='Spots')
            if spots_id and spots_id.parent.name == "p":
                spots_id = spots_id.parent
                spots_id.span.string = str(int(row['Spots']))

            #Code changes (just name of dataset)
            code_dataset_name = row['Name'].lower().replace(' ', '_')
            code_id = base_html_soup.find("span", string='"abalo_human_squamous_cell_carcinoma"')
            code_id.string = f'"{code_dataset_name}"'

            #Edit Leaderboard
            leaderboard_info = leaderboard_dataframe(row)
            html_leaderboard = dataframe_2_hmtl_table(leaderboard_info)

            tbody_id = base_html_soup.find("tbody")
            tbody_id.replace_with(BeautifulSoup(html_leaderboard.replace('\n', ''), 'html.parser'))

            #Replace all the AHSCC for the new abbreviation, this is for images paths
            base_html_str = str(base_html_soup)
            base_html_str = base_html_str.replace('AHSCC', row['Abbreviation'])

            #Fancy replace for delta symbol
            base_html_str = base_html_str.replace('&amp;', '&')

            with open(os.path.join("Datasets_Pages", row['Abbreviation'], f"dataset_{row['Abbreviation']}.html"), 'w', encoding='utf-8') as file:
                file.write(base_html_str)
        

df = load_csv_file(os.path.join('HTML Generator', 'Datasets_all_data.csv'))
create_folders(df, False)
#Sample HTML
html = load_HTML(os.path.join('Datasets_Pages', 'AHSCC', 'dataset_AHSCC.html'))
edit_HTML(html, df)
