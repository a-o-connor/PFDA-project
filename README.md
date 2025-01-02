# Programming for Data Analytics Project 
## Chemometric Analysis of Raman Spectroscopy
**Author: A O'Connor**
*****
<div align="center">
    <img src=".\img\igg1_cell_structure.jpg" alt="Cells">
</div>

## About this project
This repository contains a Python script and a Jupyter notebook with analysis of a Raman spectroscopy data set. The contents of this repository demonstrate the application of [**chemometric techniques**](https://en.wikipedia.org/wiki/Chemometrics) through Python to analyse Raman spectroscopy data collected during a Tangential Flow Filtration (TFF) process.The analysis was completed for the purposes of the *Programming for Data Analytics* module I am taking as part of a Higher Diploma in Computer Science and Data Analytics at ATU. The aim of this project is to demonstrate competency using Python to perform data analysis on large data sets.   
## Contents
### 1. Data Folder
- The data folder contains the experimental Raman spectroscopy data gathered during the TFF process.
- Reference Raman spectroscopy data used for training the model is also saved in the data folder.
- Spectroscopy files are collected from the instrument in ``.csv`` format. 
### 2. `chemometrics_analysis_script.py` Python Script
- The script includes all of the plotting and analysis functions used throughout the chemometric analysis workflow. 
### 3. `chemometric_analysis_notebook.ipnyb` Jupyter Notebook
- The notebook contains the complete Raman data analysis workflow, using the functions from the imported `chemometrics_analysis_script.py` python script to perform the various data handling and manipulation, data visualisation,preprocessing, and statistical modeling tasks. 
## Getting Started
### Dependencies
- The required Python dependencies are listed in `requirements.txt`.
- Install dependencies using:
    ````bash
    pip install -r requirements.txt
    ````
- The Jupyter notebook can be opened and executed directly, or opened in Google Colab by clicking on the link below:
<div align="center">
    <a target="_blank" href="https://colab.research.google.com/github/a-o-connor/PFDA-project/blob/main/big_project_practice.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</div>