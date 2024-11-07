# import necessary packages
## packages for handling file
import os
import shutil
## packages for handling data
import pandas as pd
## handling environment variables
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# create global variables
path_full = os.environ.get("ORI_PATH")
path_target = os.path.join(path_full, "datasets", "preprocessed")
metadata = pd.read_csv(os.path.join(path_full, "data", "raw_metadata.csv"))

# Split the fundus and oct images using raw metadata
## devide the data into fundus and oct images
fundus_imgs = metadata.loc[metadata.img_type == 'fundus',
                            ['id', 'label', 'eye_side', 'path']]
oct_imgs = metadata.loc[metadata.img_type == 'oct',
                            ['id', 'label', 'eye_side', 'path']]
labels = list(metadata.label.unique())

## create directories for the restructured dataset
for label in labels:
    os.makedirs(os.path.join(path_target, "fundus_image", label), exist_ok=True)
    os.makedirs(os.path.join(path_target, "oct_image", label), exist_ok=True)

## temporary dataframe for storing the new file name rules
new_file_name = pd.DataFrame(columns=['id', 'file_name', 'new_path'])

## restructuring the dataset
for label in labels:
    ## get the data for each label
    fundus_imgs_label = fundus_imgs.loc[fundus_imgs.label == label]
    oct_imgs_label = oct_imgs.loc[oct_imgs.label == label]
    
    ## mapping the value
    if label == labels[0]: # glaucoma == 1
        label_int = 1
    elif label == labels[1]: # non_glaucoma == 0
        label_int = 0

    ## copy the files to the new directories
    ### handle the fundus images
    for _, row in fundus_imgs_label.iterrows():
        file_name = f"fff_{label_int}_{row.id}_{row.eye_side}_{row.path[-5].lower()}.jpg"
        new_path = os.path.join(path_target, "fundus_image", label, file_name)
        try:
            shutil.copy(row.path,
                        new_path)
            new_file_name.loc[len(new_file_name)] = [row.id, file_name, new_path]
        except FileExistsError:
            print(f"File {file_name} already exists")

    ### handle the oct images
    for _, row in oct_imgs_label.iterrows():
        file_name = f"oct_{label_int}_{row.id}_{row.eye_side}_{row.path[-5].lower()}.jpg"
        new_path = os.path.join(path_target, "oct_image", label, file_name)
        try:
            shutil.copy(row.path,
                        new_path)
            new_file_name.loc[len(new_file_name)] = [row.id, file_name, new_path]
        except FileExistsError:
            print(f"File {file_name} already exists")

## save the new file name rules
new_file_name.to_csv(os.path.join(path_full, "data/refactored_metadata.csv"), index=False)

print("completed refactoring dataset.")