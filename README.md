# Glaucoma Segmentation Project
This project is a research about glaucoma detection using semantic segmentation. The fundus image and OCT scan is taken from one of Hospital in Yogyakarta. The dataset consist of fundus images labelled as glaucoma and non-glaucoma and is scanned from both patient eye.

## Confidentiality
The dataset being used is confidential because of it's medical properties. The data and metadata is not showed on the script. Also in order to download the dataset, the direct use of storage link is avoided. To handle this condition, the .env file is used to store all the confidential information that need to be used on the scripts.

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
    - IoU
    - Dice Score
    - Precision
    - Recall

## Scripts Explanation
There are several things to do before the script could be run. Also, in order to replicate the project, the script should be runned in specific order. Inside the `scripts` folder, there are guides to help in form of README file.

#### Environment
This project use 2 virtual environment for easy use.
- **General Environment** <br/>
  This environment is used during the segmentation and preparation step is being worked. The requirement file is [`requirement.txt`](https://github.com/Bugi-Sulistiyo/Glaucoma-segmentation/blob/main/requirement.txt).
- **Modeling Environment** <br/>
  This environment is used specifically on modeling and support the use of GPU during training and inference. The requirement file is [`modeling_requirement.txt`](https://github.com/Bugi-Sulistiyo/Glaucoma-segmentation/blob/main/modeling_requirement.txt). This virtual environment is made using miniconda.

## Contributor
* Bugi Sulistiyo
* Eko Rahmat Darmawan
* Anindita Septiarini
* Nur Khomah Fatmawati
* Hamdani