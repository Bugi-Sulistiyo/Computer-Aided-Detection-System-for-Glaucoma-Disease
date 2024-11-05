# import packages
## handling environment variables
import os
from dotenv import load_dotenv
## download files from google drive
from googledriver import download_folder

# load environment variables
load_dotenv()

# variables for dataset download
dataset_link = os.environ.get("DATASET_GDRIVE_LINK")    # google drive link for the dataset
folder_storage = "./datasets/original_source"            # folder to store the downloaded dataset

# download the dataset using the google drive link
download_folder(dataset_link, folder_storage)

# print the completion of the download
print("completed download dataset.")