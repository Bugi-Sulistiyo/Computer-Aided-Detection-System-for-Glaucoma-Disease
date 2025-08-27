# handling the file system
import os
# handling the evironment variable
from dotenv import load_dotenv

# load the .env file
load_dotenv()

path_full = os.environ.get("ORI_PATH")
path_download = os.path.join(path_full, "manual_download")
path_dataset = os.path.join(path_full, "datasets/preprocessed")
path_docs = os.path.join(path_full, "data")
path_zip_files = os.path.join(path_download, "zipped_files_annotation")
path_target_annot = os.path.join(path_dataset, "annotations")
path_src_imgs = os.path.join(path_dataset, "fundus_image")