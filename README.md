# Glaucoma Segmentation Project
This project is a research about glaucoma detection using SMEG dataset. The image is taken from Yoyakarta Hospital. The dataset consist of fundus images and OCT scan images.

## Confidentiality
the dataset being used is confidential. So avoid showing the dataset content in the script and avoid mention the link directly. please use .env file to store all the confidential information that need to be used on the scripts.

## Scripts Explanation
in order to do this project correctly, the following script must be run in order.
1. dataset_download.py <br>
   Download the dataset from the given google drive link. The link is stored on .env file as DATASET_GDRIVE_LINK variable.
2. create_metadata.py <br>
   Creating the metadata of the dataset from the downloaded data.
3. restructure_files.ipynb <br>
   Resctructure the file by splitting the fundus image and oct image into their own directory.

## Contributor
* Prof. Anindita Septiarini
* Bugi Sulistiyo
* Eko Rahmat Darmawan