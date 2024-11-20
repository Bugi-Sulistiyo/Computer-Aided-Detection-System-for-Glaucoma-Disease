# Scripts Explanation
In order to replicate this project, the scripts on this folder should be run according to the alphabet order of it's name.

### [`a. handle_annotation.ipynb`](a.%20handle_annotation.ipynb)
Extract the annotation from CVAT. Transform the .xml file into .json files. Also create the image with annotation on

### [`b. handle_mask.py`](b.%20handle_mask.py)
Refactor the mask image from CVAT into a more manageable structure.

### [`c. filter_used_images.py`](c.%20filter_used_images.py)
Filter the fundus and mask images and stored it into a new directory. The used images are the one who have two label (disc and cup).

### [`d. create_mask.py`](d.%20create_mask.py)
Create the mask image from each images annotations. This is being done because mask image from CVAT does not have the consistent mask information.

### [`e. dataset_cdr.ipynb`](e.%20dataset_cdr.ipynb)
Get the CDR value from the given annotion on the dataset.

### [`f. publish_dataset.ipynb`](f.%20publish_datasets.ipynb)
Process and structurize the dataset to make it proper. Also publish the dataset into hugging face hub.