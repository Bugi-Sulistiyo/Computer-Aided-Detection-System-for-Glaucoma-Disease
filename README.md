# Computer-Aided Detection System (CAD) of Glaucoma Disease
A Research of glaucoma detection using semantic segmentation. The dataset used are fundus image and OCT image taken from one of Hospital in Samarinda. The dataset consist of fundus images labelled as glaucoma and non-glaucoma and is scanned from both patient eye.

## Confidentiality
The dataset being used is confidential because of it's medical properties. The data information of each sample is embedded into the name of the files. Also in order to download the dataset, the direct use of storage link is avoided. To handle this condition, the .env file is used to store all the confidential information that needed to be used on the scripts.

## Methods and Tools Used
- Segmentation
  - **CVAT** is the main tools for annotate the images
- Modelling
  - Preprocessing
    - CLAHE augmentation
  - Model
    - U-net
    - MobileNet
    - EfficientNet
  - Evaluation Metrics
    - Train and Evalution Model
      - AUC
      - Precision
      - Recall
      - Mean px Accuracy
    - Segmentation Model
      - Huber
      - MSE
      - MAE

## How to Replicate the Project
There are several things to do before the script could be run. Also, in order to replicate the project, the script should be runned in specific order (alphabetical or numerical). Inside the `scripts` folder, there are guides to help in form of markdown.

#### Environment
In order for Tensorflow use GPU when training or inference, it recommended to use venv from miniconda. Following guide from [Install Tensorflow with pip](https://www.tensorflow.org/install/pip#windows-native). All the necessary dependencies is stored on [`requirements.txt`](requirements.txt) file.<br>
To make there's no error caused by path and the keep the confidentiality of the dataset, the `.env` file is used. there are two variables in it such as:
- DATASET_GDRIVE_LINK → store the dataset link
- ORI_PATH → store the full path of where the project is stored

## Contributor
* Bugi Sulistiyo
* Eko Rahmat Darmawan
* Anindita Septiarini
* Nur Khomah Fatmawati
* Hamdani