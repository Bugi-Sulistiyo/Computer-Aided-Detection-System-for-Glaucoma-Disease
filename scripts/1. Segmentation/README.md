# Scripts Explanation
In order to replicate this project, the scripts on this folder should be run according to the alphabet order of it's name.

### [`a. dataset_download.py`](a.%20dataset_download.py)
This script intended for download the SMEC dataset from google drive. The google drive link is stored on `.env` files.

### [`b. create_metadata.py`](b.%20create_metadata.py)
This script intended for creating the metadata of the dataset. Also create a unique code to hide the patient information but yet, still trackable when needed. Additionally, this scripts map which image is fundus and which one is OCT scan image. The metadata created is saved in form of .csv file.

### [`c. restructure_files.ipynb`](c.%20refactor_dataset.py)
This script intended to refactor the dataset structure. The OCT scan image and fundus image is stored into a new directory with a new file name format. The naming convention format could be seen on this [docs](https://docs.google.com/document/d/e/2PACX-1vQemzjSsW43qPY2vZWeov_9snclNJvQHRirW1xNL0lDpN5uzwBLnyWYG8VX8xyY5ABAsKqxHbvbYLjY/pub).

### [`d. error.ipynb`](d.%20error.ipynb)
This script is intended to solve the error existing on the dataset or caused by another scripts.

### [`e. analsis.ipynb`](e.%20analysis.ipynb)
This script is intended to analyse more information from the dataset using the existing metadata