# import packages
import os
from dotenv import load_dotenv

from googledriver import download_folder

# load environment variables
load_dotenv()

# variables for dataset download
dataset_link = os.environ.get("DATASET_GDRIVE_LINK")
folder_storage = "./dataset/"

# download the dataset using the google drive link
download_folder(dataset_link, folder_storage)

# print the completion of the download
print("completed download dataset.")