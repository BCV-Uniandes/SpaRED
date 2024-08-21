# Library_Spared_Spackle

This repository contains all the necessary files to create a PyPI library to the SPARED and SpaCKLE contributions

This is the  README file which will contain the long description of the PiPy library. Most libraries have a README file. Mean while this file will only contain this information and will be soon updated. 

## Enhancing Gene Expression Prediction from Histology Images with Spatial Transcriptomics Completion

[Gabriel Mejía](https://scholar.google.com/citations?hl=es&user=yh69hnYAAAAJ)<sup>1,2</sup>\*, [Daniela Ruiz](https://scholar.google.com/citations?hl=es&user=Zm-tYR0AAAAJ)<sup>1,2</sup>\*, Paula Cárdenas<sup>1,2</sup>, Leonardo Manrique<sup>1,2</sup>, Daniela Vega<sup>1,2</sup>, [Pablo Arbelaez](https://scholar.google.com/citations?hl=es&user=k0nZO90AAAAJ)<sup>1,2</sup>

<br/>
<font size="1"><sup>*</sup>Equal contribution.</font><br/>
<font size="1"><sup>1 </sup> Center  for  Research  and  Formation  in  Artificial  Intelligence (<a href="https://cinfonia.uniandes.edu.co">CinfonIA</a>), Bogotá, Colombia.</font><br/>
<font size="1"><sup>2 </sup> Universidad  de  los  Andes,  Bogotá, Colombia.</font><br/>

- Preprint available at arXiv
- Visit the project on our [website](https://bcv-uniandes.github.io/spared_webpage/)

### Abstract

Spatial Transcriptomics is a novel technology that aligns histology images with spatially resolved gene expression profiles. Although groundbreaking, it struggles with gene capture yielding high corruption in acquired data. Given potential applications, recent efforts have focused on predicting transcriptomic profiles solely from histology images. However, differences in databases, preprocessing techniques, and training hyperparameters impact a fair comparison between methods. To address these challenges, we present a systematically curated and processed database collected from 26 public sources, representing an 8.6-fold increase compared to previous works. Additionally, we propose a state-of-the-art transformer-based completion technique for inferring gene expression, which significantly boosts the performance of transcriptomic profile predictions across all datasets. Altogether, our contributions constitute the most comprehensive benchmark of gene expression prediction from histology images to date and a stepping stone for future research.

## System Dependencies

Before installing the Python package, ensure the following system dependencies are installed:

```shell
conda create -n spared
conda activate spared
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
pip install torch_geometric
conda install -c conda-forge squidpy
pip install wandb
pip install wget
pip install combat
pip install opencv-python
pip install positional-encodings[pytorch]
pip install openpyxl
pip install pyzipper
pip install plotly
pip install sh
pip install sphinx
pip install -U sphinx-copybutton
pip install -U sphinx_rtd_theme
```
