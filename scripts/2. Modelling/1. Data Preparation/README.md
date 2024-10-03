# Scripts Explanation
In order to replicate this project, the scripts on this folder should be run according to the alphabet order of it's name.

### `a. handle_annotation.ipynb`
Extract the annotation from CVAT. Transform the .xml file into .json files.

### `b. handle_mask.ipynb`
Refactor the mask image from CVAT into a more manageable structure.

### `c. filter_used_images.ipynb`
Filter the fundus and mask images into a new directory. The used images are the one who have two label (disc and cup).

### `d. create_mask.ipynb`
Create the mask image from each images annotations. This is being done because mask image from CVAT does not have the consistent mask information.

### `e. remove_corrupt.ipynb`
Remove the fundus and mask images that is corrupt.