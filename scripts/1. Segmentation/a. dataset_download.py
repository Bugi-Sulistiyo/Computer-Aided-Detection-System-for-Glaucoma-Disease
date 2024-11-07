# import packages
## handling file
import os
## handling environment variables
from dotenv import load_dotenv
## download files from google drive
from googledriver import download_folder

# load environment variables
load_dotenv()

# variables for dataset download
dataset_link = os.environ.get("DATASET_GDRIVE_LINK")                    # google drive link for the dataset
path_full = os.environ.get("ORI_PATH")                                  # path to the project folder
folder_storage = os.path.join(path_full, "datasets/original_source")    # folder to store the downloaded dataset

# download the dataset using the google drive link
download_folder(dataset_link, folder_storage)

# print the completion of the download
print("completed download dataset.")