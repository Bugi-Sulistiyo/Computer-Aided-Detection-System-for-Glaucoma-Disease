import tensorflow as tf
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt
import numpy as np

from .viz_utils import visualize_bounding_box

def count_dataset_cdr(mask_path:str, visualize:bool=False):
    """Count the CDR value from the mask image

    Args:
        mask_path (str): the path of the mask image
        visualize (bool, optional): visualize the mask image with annotation. Defaults to False.

    Returns:
        Dict: the CDR values (Area CDR, Horizontal CDR, Vertical CDR)
    """
    # read the mask file
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (512, 512), method="nearest")
    mask = tf.cast(mask, tf.int32)

    # change the mask image into 2D
    mask_2d = mask[:,:,0]
    # get the bounding box of the mask
    cup_mask = mask_2d == 64
    disc_mask = mask_2d == 255
    cup_bbox = regionprops(label(cup_mask))[0].bbox
    disc_bbox = regionprops(label(disc_mask))[0].bbox

    # calculate the CDR variables
    cup_width = cup_bbox[3] - cup_bbox[1]
    disc_width = disc_bbox[3] - disc_bbox[1]
    cup_height = cup_bbox[2] - cup_bbox[0]
    disc_height = disc_bbox[2] - disc_bbox[0]

    # visualize the mask with bounding box
    if visualize:
        plt.figure(figsize=(10, 10))

        plt.subplot(2,2, 1)
        plt.imshow(mask_2d, cmap="gray")
        plt.gca().add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),
                                        cup_width, cup_height,
                                        edgecolor='red', facecolor='none'))
        plt.gca().add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),
                                        disc_width, disc_height,
                                        edgecolor='cyan', facecolor='none'))
        plt.title("Original Mask")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(cup_mask, cmap="gray")
        plt.gca().add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),
                                        cup_width, cup_height,
                                        edgecolor='r', facecolor='none'))
        plt.title("Cup Mask")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(disc_mask, cmap="gray")
        plt.gca().add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),
                                        disc_width, disc_height,
                                        edgecolor='c', facecolor='none'))
        plt.title("Disc Mask")
        plt.axis("off")
        plt.show()
    # return the CDR values
    return {"Area CDR": np.sum(cup_mask) / np.sum(np.logical_or(disc_mask, cup_mask)),
            "Horizontal CDR": cup_width / disc_width,
            "Vertical CDR": cup_height / disc_height}

def split_disc_cup_mask(pred_mask, treshold:float=0.1, img_idx:int=13, visualize:bool=True):
    """split the disc and cup section from the predicted mask

    Args:
        pred_mask (tf.Tensor): the predicted mask from model
        treshold (float, optional): the treshold used to make mask as binary. Defaults to 0.1.
        img_idx (int, optional): the index of mask that want to be visualized. Defaults to 13.
        visualize (bool, optional): whether to visualize the result or not. Defaults to True.

    Returns:
        tf.Tensor: the result of the splitted mask based on the label
    """
    # devide the mask into two separate mask
    cup_mask = pred_mask[..., 1]
    disc_mask = pred_mask[..., 2]

    # transform the mask image into a binary mask image
    binary_cup_mask = tf.where(cup_mask > treshold, 1, 0)
    binary_disc_mask = tf.where(disc_mask > treshold, 1, 0)

    if not visualize:
        return cup_mask, disc_mask, binary_cup_mask, binary_disc_mask

    # show the predicted mask image directly
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.title("Predicted Cup")
    plt.imshow(cup_mask[img_idx], cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("Predicted Disc")
    plt.imshow(disc_mask[img_idx], cmap="gray")

    # show the binary mask image
    plt.subplot(2, 2, 3)
    plt.title("Binary Cup")
    plt.imshow(binary_cup_mask[img_idx], cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("Binary Disc")
    plt.imshow(binary_disc_mask[img_idx], cmap="gray")

    plt.show()

    return cup_mask, disc_mask, binary_cup_mask, binary_disc_mask

def get_bounding_box(mask:np.array):
    """get the bounding box of the mask

    Args:
        mask (np.array): the mask of the image

    Returns:
        tuple: the bounding box of the mask and the size of the mask (ymin, ymax, xmin, xmax, height, width)
    """

    # split the mask into rows and columns
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # get the bounding box of the mask
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # get the size of the mask
    height = ymax - ymin + 1
    width = xmax - xmin + 1

    return ymin, ymax, xmin, xmax, height, width

def calculate_area_CDR(cup_mask:np.array, disc_mask:np.array, bcup_mask:np.array, bdisc_mask:np.array, visualize:bool=True):
    """calculate the area and CDR of the mask

    Args:
        cup_mask (np.array): the cup mask
        disc_mask (np.array): the disc mask
        bcup_mask (np.array): the binary cup mask
        bdisc_mask (np.array): the binary disc mask

    Returns:
        list[dict, dict]: the area and CDR of the mask, the bounding box of the mask
    """
    # count the area CDR
    cup_area = np.sum(bcup_mask)
    disc_area = np.sum(bdisc_mask)
    acdr = cup_area / disc_area

    d_ymin, d_ymax, d_xmin, d_xmax, d_height, d_width = get_bounding_box(bdisc_mask)
    c_ymin, c_ymax, c_xmin, c_xmax, c_height, c_width = get_bounding_box(bcup_mask)

    # count the horizontal CDR
    h_cdr = c_width / d_width
    # count the vertical CDR
    v_cdr = c_height / d_height

    if visualize:
        visualize_bounding_box("Disc", disc_mask, bdisc_mask, d_ymin, d_ymax, d_xmin, d_xmax)
        visualize_bounding_box("Cup", cup_mask, bcup_mask, c_ymin, c_ymax, c_xmin, c_xmax)

    return [{"cup_area": cup_area,
            "disc_area": disc_area,
            "acdr": acdr,
            "h_cdr": h_cdr,
            "v_cdr": v_cdr},
            {"d_ymin": d_ymin,
            "d_ymax": d_ymax,
            "d_xmin": d_xmin,
            "d_xmax": d_xmax,
            "d_height": d_height,
            "d_width": d_width,
            "c_ymin": c_ymin,
            "c_ymax": c_ymax,
            "c_xmin": c_xmin,
            "c_xmax": c_xmax,
            "c_height": c_height,
            "c_width": c_width}]