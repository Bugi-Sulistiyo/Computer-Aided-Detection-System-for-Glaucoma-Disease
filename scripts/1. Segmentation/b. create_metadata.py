# import packages
## package for handling file
import os
import re
## package for handling data
import pandas as pd
import numpy as np
## package for handling image
import cv2
## package for handling environment variables
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# create global variables
path_full = os.environ.get("ORI_PATH")
metadata = pd.DataFrame(columns=["label", "patient", "id", "gender",
                                "img_type", "eye_side", "path"])
labels = ["glaucoma", "non_glaucoma"]
data_path = os.path.join(path_full, "datasets/original_source")
metadata_path = os.path.join(path_full, "data")

# create the metadata
for index, label_src in enumerate(os.listdir(data_path)):
    path_label = os.path.join(data_path, label_src)

    ## rename label folder
    if path_label != os.path.join(data_path, labels[index]):
        os.rename(path_label,
                os.path.join(data_path, labels[index]))
    
    for patient in os.listdir(os.path.join(data_path,
                                            labels[index])):
        path_patient = os.path.join(data_path,
                                    labels[index],
                                    patient)
        
        ## getting metadata information
        ### get patient name
        patient_clean = patient.replace("(", "").replace(")", "")
        name = re.sub(r'\d+', '',
                        patient_clean).lower().replace(" $", "").replace("tn.", "").replace("nn.", "").replace("ny.", "").replace("nn", "").replace("ny", "").replace("tn", "")
        ### get patient id
        id = re.findall(r'\d+', patient_clean)[0]
        ### get patient gender
        if ((re.search("nn", patient_clean.lower()) != None)
            or (re.search("ny", patient_clean.lower()) != None)):
            gender = "woman"
        elif re.search("tn", patient_clean.lower()) != None:
            gender = "man"
        else:
            gender = "unknown"
        
        ### filter the image files
        for file_name in os.listdir(path_patient):
            if ((file_name[-3:] != "jpg")
                and (file_name[-3:] != "png")
                and (file_name[-3:] != "peg")):
                continue

            final_path = os.path.join(path_patient, file_name)

            ### get patient eye side
            if re.search("od", file_name.lower()) != None:
                eye_side = "r" # for right eye OD == right
            elif re.search("os", file_name.lower()) != None:
                eye_side = "l" # for left eye OS == left
            else:
                eye_side = "u" # for others (expected: for oct images)
            ### get image type
            img = cv2.imread(final_path)
            white_px = np.sum(img == 255)
            black_px = np.sum(img == 0)
            if white_px > black_px:
                img_type = "oct"
            else:
                img_type = "fundus"
            
            ## add metadata of each patient image to the dataframe
            metadata.loc[len(metadata.index)] = [labels[index], name.strip(),
                                                id, gender,
                                                img_type, eye_side,
                                                final_path]

# save metadata to csv
if not os.path.exists(metadata_path):
    os.makedirs(metadata_path)
metadata.to_csv(os.path.join(metadata_path, "raw_metadata.csv"), index=False)

print("completed create metadata.")