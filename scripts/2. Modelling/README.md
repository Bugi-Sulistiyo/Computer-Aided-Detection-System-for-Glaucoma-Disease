# Flow
The modelling part is devided into:
1. Preparation <br/>
   The scripts on this section is handling the preparation of needed before the preprocessing and modeling could be started. The scripts in here mainly create an images, refactor it, get the needed properties to be used on the next steps.
2. Preprocessing <br/>
   The scripts on this section is handling the pre-processing implemented to support the modeling. The scripts is handling the implementation of augmentation and data splitting
3. Modelling <br/>
   The scripts on this section handle the training, inference, post-processing things.

## Scripts Explanation
- [`utilities.py`](utilities.py) <br>
   This scripts store all the function needed to on all stages of modeling