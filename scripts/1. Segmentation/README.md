# Scripts Explanation
In order to replicate this project, the scripts on this folder should be run according to the alphabet order of it's name.

### `a. dataset_download.py`
This script intended for download the SMEC dataset from google drive. The google drive link is stored on .env files

### `b. create_metadata.py`
This script intended for creating the dataset metadata. Also create a unique code to hide the patient information but yet, still trackable when needed. Additionally, this scripts map which image is fundus and which one is OCT scan image. The metadata created is saved in form of .csv file.

### `c. restructure_files.ipynb`
This script intended to refactor the dataset structure. The OCT scan image and fundus image is stored into a new directory with a new file name format. The naming convention format could be seen on this [docs](https://docs.google.com/document/d/e/2PACX-1vQemzjSsW43qPY2vZWeov_9snclNJvQHRirW1xNL0lDpN5uzwBLnyWYG8VX8xyY5ABAsKqxHbvbYLjY/pub).

### `d. handle_duplicates.ipynb`
After running the script `b` and `c`, there will be a bug. The image being copied is less than the original dataset. there for, to solve that problem this script exist to solve the problem. After running this script, the `b. create_metadata.py` and `c. restructure_files.ipynb` scripts should be rerunned.

### `e. patient_with_more_fundus.ipynb`
Getting the information of how many fundus image is taken from a single eye on every patient.

### `f. image_per_eye.ipynb`
Get the information of the average image per eye.

### `g. create_annot_img.ipynb`
Create fundus image with annotaion of each label is drawn on it.