# Import the needed package
## package for handing files
import os
## package for handling images
import numpy as np
import cv2
from PIL import Image
## package for handling json files
import json
from dotenv import load_dotenv

load_dotenv()

# Global Variables
path_full = os.environ.get("ORI_PATH")
path_main = os.path.join(path_full, "datasets", "preprocessed")
path_img = os.path.join(path_main, "fundus_image")
path_anot = os.path.join(path_main, "annotations")
path_mask = os.path.join(path_full, "datasets", "cleaned", "mask_image")
classes = ["glaucoma", "non_glaucoma"]

# Get the annotation information
annots = {}
for class_type in classes:
    annots[class_type] = json.load(open([os.path.join(path_anot,
                                                        class_type,
                                                        file)
                                            for file in os.listdir(os.path.join(path_anot,
                                                                                class_type))
                                            if file.endswith(".json")][0]))

# Create the mask image
# loop for each classes (glaucoma, non_glaucoma)
for class_type, annotations in annots.items():
    # loop for every image in the class
    for annotation in annotations:
        # make sure the image have disc and cup annotation
        if len(annotation["annotation"]) < 2:
            continue
        # create a black image with the same size as the image
        b_mask = np.zeros((int(annotation["metadata"]["img_height"]),
                            int(annotation["metadata"]["img_width"])), dtype=np.uint8)
        
        # draw the white space (mask area)
        for polygon in sorted(annotation["annotation"], key=lambda x: x["label"], reverse=True):
            # define the color of the mask
            label = 255 if polygon["label"] == "disc" else 64
            # get points of the polygon
            points = [list(map(np.float32, point.split(","))) for point in polygon["points"].split(";")]
            # fill the polygon with the color
            cv2.fillPoly(b_mask, [np.array(points, dtype=np.int32)], color=label)
        # check the image already exist or not
        try:
            os.remove(os.path.join(path_mask, f"{annotation['metadata']['img_name'].split('.')[0]}.png"))
        except FileNotFoundError:
            continue
        # save the mask image
        Image.fromarray(b_mask).save(os.path.join(path_mask, f"{annotation['metadata']['img_name'].split('.')[0]}_mask.png"))

# show the success message
print("New Mask image has been created")