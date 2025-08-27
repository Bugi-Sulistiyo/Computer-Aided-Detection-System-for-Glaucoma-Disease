# Import the necessary package
import sys                      # handling system arguments
# add path to utilities directory
sys.path.insert(0, "./../../utilities")

import os                       # handling file for basic operation
import shutil                   # handling file for duplication
import zipfile as zf            # handling the zipped file
import paths                    # handling the path

# Initialize the global variables
classes = ["glaucoma", "non_glaucoma"]
zip_files = [file for file in os.listdir(paths.path_zip_mask) if file.endswith(".zip")]

# Create the directory for the mask images
## create directories for temporary storage of unzipped files
for file in zip_files:
    new_dir = os.path.join(paths.path_dataset_preprocessed, "temp_mask_unzip", file.split(".")[0])
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        print(f"Directory {new_dir} already exists")

## create directories for mask images
for class_name in classes:
    new_dir = os.path.join(paths.path_dataset_preprocessed, "mask_image", class_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        print(f"Directory {new_dir} already exists")

del file, new_dir, class_name

# Extract the zip files
for zip_file in zip_files:
    with zf.ZipFile(os.path.join(paths.path_zip_mask, zip_file), "r") as zip_ref:
        zip_ref.extractall(os.path.join(paths.path_dataset_preprocessed, "temp_mask_unzip", zip_file.split(".")[0]))
del zip_file, zip_ref

# Merge the mask images
for directory in zip_files:
    directory = directory.split(".")[0]
    for class_name in classes:
        src_directory = os.path.join(paths.path_dataset_preprocessed, "temp_mask_unzip", directory, "SegmentationClass")
        dst_directory = os.path.join(paths.path_dataset_preprocessed, "mask_image", class_name)
        if class_name.split("_")[0] in directory.split("_")[1]:
            for file in os.listdir(src_directory):
                if file.endswith(".png") and file.split(".")[0] != "fff_0_122451_l_1":
                    shutil.copy(os.path.join(src_directory, file), dst_directory)
del directory, src_directory, dst_directory, file, class_name

# Remove the temporary directories
shutil.rmtree(os.path.join(paths.path_dataset_preprocessed, "temp_mask_unzip"))

# Print the success message
print("Mask images from CVAT have been successfully extracted and stored in the appropriate directories")