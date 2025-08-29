# Import the needed package
import sys                      # handling system arguments
# add path to utilities directory
sys.path.insert(0, "./../../")
import os                       # handling file operation basic
import shutil                   # handling file operation for copying and removing
import json                     # handling json file
import pandas as pd             # handling dataframe for tabular data
# import the needed path
from utilities import path_prep_fundus, path_prep_mask, path_prep_annot_target, path_clean_fundus, path_clean_mask, path_docs

# Global Variables
classes = os.listdir(path_prep_fundus)

# Filter the used images
## import the metadata for each classes
gcm_meta = json.load(open(os.path.join(path_prep_annot_target, classes[0], 'annotations.json')))
ngcm_meta = json.load(open(os.path.join(path_prep_annot_target, classes[1], 'annotations.json')))

used_img_gcm = []
used_img_ngcm = []

## get the list of images that have 2 annotations
for image_meta in gcm_meta:
    if len(image_meta['annotation']) == 2:
        used_img_gcm.append(image_meta['metadata']['img_name'])
for image_meta in ngcm_meta:
    if len(image_meta['annotation']) == 2:
        used_img_ngcm.append(image_meta['metadata']['img_name'])

# Store the filtering summary result
## define the dataframe structure
filter_result = pd.DataFrame(columns=['class', 'used', 'deprecated', 'total', 'used_percentage', 'deprecated_percentage'])
## fill the dataframe
filter_result.loc[len(filter_result)] = ['glaucoma',
                                        len(used_img_gcm),
                                        len(gcm_meta) - len(used_img_gcm),
                                        len(gcm_meta),
                                        round(len(used_img_gcm) / len(gcm_meta) * 100, 2),
                                        round((len(gcm_meta) - len(used_img_gcm)) / len(gcm_meta) * 100, 2)]
filter_result.loc[len(filter_result)] = ['non_glaucoma',
                                        len(used_img_ngcm),
                                        len(ngcm_meta) - len(used_img_ngcm),
                                        len(ngcm_meta),
                                        round(len(used_img_ngcm) / len(ngcm_meta) * 100, 2),
                                        round((len(ngcm_meta) - len(used_img_ngcm)) / len(ngcm_meta) * 100, 2)]
filter_result.loc[len(filter_result)] = ['total',
                                        len(used_img_gcm) + len(used_img_ngcm),
                                        len(gcm_meta) + len(ngcm_meta) - len(used_img_gcm) - len(used_img_ngcm),
                                        len(gcm_meta) + len(ngcm_meta),
                                        round((len(used_img_gcm) + len(used_img_ngcm)) / (len(gcm_meta) + len(ngcm_meta)) * 100, 2),
                                        round((len(gcm_meta) + len(ngcm_meta) - len(used_img_gcm) - len(used_img_ngcm)) / (len(gcm_meta) + len(ngcm_meta)) * 100, 2)]
## save the dataframe
filter_result.to_csv(os.path.join(path_docs, 'filter_result.csv'), index=False)

# Create the destination directories
for new_dir in [path_clean_fundus, path_clean_mask]:
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        print(f'{new_dir} already exists')

# Copy the used images
for class_name, merged_img in {classes[0]:used_img_gcm, classes[1]:used_img_ngcm}.items():
    for img in merged_img:
        ## copy the fundus images
        src_fundus_path = os.path.join(path_prep_fundus, class_name, img)
        dst_fundus_path = os.path.join(path_clean_fundus, img)
        try:
            shutil.copy(src_fundus_path, dst_fundus_path)
        except FileNotFoundError:
            print(f'{src_fundus_path} not found')
        except shutil.Error:
            print(f'{dst_fundus_path} already exists')
        ## copy the mask images
        src_mask_path = os.path.join(path_prep_mask, class_name, img.replace('.jpg', '.png'))
        dst_mask_path = os.path.join(path_clean_mask, img.replace('.jpg', '.png'))
        try:
            shutil.copy(src_mask_path, dst_mask_path)
        except FileNotFoundError:
            print(f'{src_mask_path} not found')
        except shutil.Error:
            print(f'{dst_fundus_path} already exists')

# Show the success message
print('Filtering the used is done')