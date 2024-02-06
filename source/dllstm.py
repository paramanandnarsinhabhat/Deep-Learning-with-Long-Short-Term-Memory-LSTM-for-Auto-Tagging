
'''
1. Load Data and Import Libraries
2. Text Cleaning
3. Merge Tags with Questions
4. Dataset Prepartion
5. Text Representation
6. Model Building
    1. Define Model Architecture
    2. Train the Model
7. Model Predictions
8. Model Evaluation
9. Inference
'''



# Load Data and Import Libraries
import re 

#reading files
import pandas as pd

#handling html data
from bs4 import BeautifulSoup

#visualization
import matplotlib.pyplot as plt  


pd.set_option('display.max_colwidth', 200)


import zipfile
import os

# Specify the path to the zip file
zip_file_path = 'data/archive (2).zip'

# Specify the directory to extract to
extract_to_dir = 'data/unzipped_contents'

# Create a directory to extract to if it doesn't exist
os.makedirs(extract_to_dir, exist_ok=True)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(extract_to_dir)
    
    # List the contents of the extracted folder
    print(f"Contents of the zip file '{zip_file_path}':")
    for file_name in zip_ref.namelist():
        print(file_name)

# Now you can access files inside the unzipped directory
# For example, to open a file:
# with open(os.path.join(extract_to_dir, 'yourfile.txt'), 'r') as file:
#     print(file.read())
        
        