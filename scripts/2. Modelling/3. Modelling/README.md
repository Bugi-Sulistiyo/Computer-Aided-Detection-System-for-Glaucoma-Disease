# Scripts Explanation
In order to replicate this project, the scripts on this folder should be run according to the alphabet order of it's name.

### [`a. training.ipynb`](a.%20training.ipynb)
Create, train, and evaluate the model

### [`b. visualize_result.ipynb`](b.%20visualize_result.ipynb)
Test the model and check the model result by visualizing the predicted mask.

### [`c. classify_glaucoma.ipynb`](c.%20classify_glaucoma.ipynb)
Find the CDR value from predicted mask and set the treshold to classify the glaucoma.

### [`d. evaluate_segmentation.ipynb`](d.%20evaluate_segmentation.ipynb)
Evaluate the model performance base on the CDR value created from the predicted mask image. Also get the accuracy from the classification of glaucoma or non-glaucoma

### [`e. train_log.ipynb`](e.%20train_log.ipynb)
Get the train log data from wandb