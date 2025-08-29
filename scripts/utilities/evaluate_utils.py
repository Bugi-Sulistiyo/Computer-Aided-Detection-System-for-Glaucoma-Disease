import tensorflow as tf
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
import pandas as pd

def ev_cdr(model:tf.keras.Model, img_path:str, mask_path:str, threshold:float=.5, img_size:int=128, visualize:bool=False):
    """calculate the CDR value of the given image with the given model

    Args:
        model (tf.keras.Model): the model that will be used
        img_path (str): the path of the test image
        mask_path (str): the path of the mask image
        threshold (float, optional): threshold define active pixel to make binary image. Defaults to .5.
        img_size (int, optional): the size of image in rasio of 1:1. Defaults to 128.
        visualize (bool, optional): also visualize the image or not. Defaults to False.

    Returns:
        dict: the CDR value
    """
    # preprocess and get the image
    img = tf.io.read_file(img_path)                                         # read the image
    img = tf.image.decode_jpeg(img, channels=3)                             # read the jpg file to readable type
    img = tf.image.resize(img, (img_size, img_size), method='nearest')      # change the image size
    img = tf.cast(img, tf.float32) / 255.                                   # normalize the image
    img = tf.expand_dims(img, axis=0)                                       # change the shape of the image to simulate batch dataset

    # predict the image
    pred_mask = model.predict(img, verbose=0)

    # split the mask image to a binary mask image
    cup_mask = tf.where(pred_mask[..., 1] > threshold, 1, 0)
    disc_mask = tf.where(pred_mask[..., 2] > threshold, 1, 0)

    # create a bounding box for each binary image (will get the coordinate)
    cup_bbox = regionprops(label(cup_mask.numpy())[0])[0].bbox
    disc_bbox = regionprops(label(disc_mask.numpy())[0])[0].bbox

    # calculate the size of bounding box
    cup_width = cup_bbox[3] - cup_bbox[1]
    cup_height = cup_bbox[2] - cup_bbox[0]
    disc_width = disc_bbox[3] - disc_bbox[1]
    disc_height = disc_bbox[2] - disc_bbox[0]

    # visualize the result
    if visualize:
        plt.figure(figsize=(10, 10))

        # show the original mask of the given fundus image
        plt.subplot(2,2, 1)
        plt.imshow(plt.imread(mask_path), cmap="gray")                      # show image
        plt.title("Original Mask")
        plt.axis("off")

        # show the predicted mask with bounding box on it
        plt.subplot(2,2, 2)
        plt.imshow(tf.argmax(pred_mask, axis=-1)[0], cmap="gray")           # show image
        plt.gca().add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),       # show bounding box of cup
                                        cup_width, cup_height,
                                        edgecolor='r', facecolor='none'))
        plt.gca().add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),     # show bounding box of disc
                                        disc_width, disc_height,
                                        edgecolor='c', facecolor='none'))
        plt.title("Predicted Mask")
        plt.axis("off")

        # show the cup binary mask with bounding box on it
        plt.subplot(2, 2, 3)
        plt.imshow(cup_mask[0], cmap="gray")                                # show image
        plt.gca().add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),       # show bounding box
                                        cup_width, cup_height,
                                        edgecolor='r', facecolor='none'))
        plt.title("Cup Mask")
        plt.axis("off")

        # show the disc binary mask with bounding box on it
        plt.subplot(2, 2, 4)
        plt.imshow(disc_mask[0], cmap="gray")                               # show image
        plt.gca().add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),     # show bounding box
                                        disc_width, disc_height,
                                        edgecolor='c', facecolor='none'))
        plt.title("Disc Mask")
        plt.axis("off")
        plt.show()
    
    # calculate the CDR value
    return {"area_cdr": np.sum(cup_mask) / np.sum(np.logical_or(disc_mask, cup_mask)),  # area CDR
            "horizontal_cdr": cup_width / disc_width,                                   # horizontal CDR
            "vertical_cdr": cup_height / disc_height}                                   # vertical CDR

def count_loss_cdr(dts_cdr:pd.core.series.Series, pred_cdr:pd.core.series.Series):
    """count the loss value of predicted CDR and dataset CDR

    Args:
        dts_cdr (pd.core.series.Series): the dataset CDR data
        pred_cdr (pd.core.series.Series): the predicetd CDR data

    Returns:
        disct: the regression loss value
    """
    # remove the added id part
    dts_cdr.id = dts_cdr.id.apply(lambda x: x.replace("_mask", ""))
    pred_cdr.id = pred_cdr.id.apply(lambda x: x.replace("_aug", ""))
    # join the the two data using inner join
    cdr_evalution = dts_cdr.merge(pred_cdr, on="id", suffixes=("_true", "_pred"))

    # initiate the loss function
    mse = MeanSquaredError()
    mae = MeanAbsoluteError()
    huber = Huber()

    # calculate the loss value for each CDR value
    return {"a_mse": mse(cdr_evalution.a_cdr_true, cdr_evalution.a_cdr_pred).numpy(),
            "a_mae": mae(cdr_evalution.a_cdr_true, cdr_evalution.a_cdr_pred).numpy(),
            "a_huber": huber(cdr_evalution.a_cdr_true, cdr_evalution.a_cdr_pred).numpy(),
            "h_mse": mse(cdr_evalution.h_cdr_true, cdr_evalution.h_cdr_pred).numpy(),
            "h_mae": mae(cdr_evalution.h_cdr_true, cdr_evalution.h_cdr_pred).numpy(),
            "h_huber": huber(cdr_evalution.h_cdr_true, cdr_evalution.h_cdr_pred).numpy(),
            "v_mse": mse(cdr_evalution.v_cdr_true, cdr_evalution.v_cdr_pred).numpy(),
            "v_mae": mae(cdr_evalution.v_cdr_true, cdr_evalution.v_cdr_pred).numpy(),
            "v_huber": huber(cdr_evalution.v_cdr_true, cdr_evalution.v_cdr_pred).numpy()}