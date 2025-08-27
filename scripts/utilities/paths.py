# handling the file system
import os
# handling the evironment variable
from dotenv import load_dotenv

# load the .env file
load_dotenv()

# main project path
path_full = os.environ.get("ORI_PATH")

# submain path for storing all main data according to it's general type
path_docs = os.path.join(path_full, "data")
path_dataset = os.path.join(path_full, "datasets")
path_download = os.path.join(path_full, "manual_download")
path_huggingface = os.path.join(path_full, "dataset_hf")

# heading 2 of path. storing the breakdown of submain path
path_dataset_cleaned = os.path.join(path_dataset, "cleaned")
path_dataset_origin = os.path.join(path_dataset, "original_source")
path_dataset_preprocessed = os.path.join(path_dataset, "preprocessed")
path_dataset_splitted = os.path.join(path_dataset, "splitted")
path_zip_annot = os.path.join(path_download, "zipped_files_annotation")
path_zip_mask = os.path.join(path_download, "zipped_files_mask")

# heading 3 of path. storing the breakdown of heading 2 path
path_target_annot = os.path.join(path_dataset_preprocessed, "annotations")
path_src_imgs = os.path.join(path_dataset_preprocessed, "fundus_image")

path_clean_aug = os.path.join(path_dataset_cleaned, "aug_image")
path_clean_fundus = os.path.join(path_dataset_cleaned, "fundus_image")
path_clean_mask = os.path.join(path_dataset_cleaned, "mask_image")
path_prep_annot_img = os.path.join(path_dataset_preprocessed, "annot_image")
path_prep_annot_target = os.path.join(path_dataset_preprocessed, "annotations")
path_prep_fundus = os.path.join(path_dataset_preprocessed, "fundus_image")
path_prep_mask = os.path.join(path_dataset_preprocessed, "mask_image")
path_prep_oct = os.path.join(path_dataset_preprocessed, "oct_image")