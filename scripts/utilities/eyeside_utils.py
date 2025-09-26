import tensorflow as tf
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def largest_region(mask:np.ndarray):
    """Get the largest region from a mask

    Args:
        mask (np.ndarray): The mask to get the largest region from

    Returns:
        RegionProperties: The largest region from the mask
    """
    # compute the label of the mask
    props = regionprops(label(mask))
    # return the largest region
    return max(props, key=lambda p: p.area) if props else None

def split_eyeside(dataset:tf.data.Dataset, model:tf.keras.Model, treshold:float=.5) -> list:
    """Split the eyeside of the dataset

    Args:
        dataset (tf.data.Dataset): The dataset to split the eyeside from
        model (tf.keras.Model): The model to use for prediction
        treshold (float, optional): The threshold to use for prediction. Defaults to .5.

    Returns:
        list: The list of the split eyeside
    """
    # Initialize the global variables
    result = []
    side_map = {"l": "left", "r": "right"}
    idx_across_batches = 0 # global index across batches

    # loop over the dataset
    for images, _, img_paths in dataset:
        # predick a mask image
        pred_mask = model.predict(images, verbose=0)
        # get the batch size
        batch_size = images.shape[0]

        # loop over the batch
        for index in range(batch_size):
            # get the filename & label
            path_str = img_paths[index].numpy().decode("utf-8")
            file_name = path_str.split("\\")[-1]
            real_label = side_map.get(file_name.split("_")[3], "unknown")
            # get the mask & image
            cup_mask = (pred_mask[index, ..., 1] > treshold).astype("uint8")
            disc_mask = (pred_mask[index, ..., 2] > treshold).astype("uint8")
            img = images[index].numpy()

            # initialize the base entry
            entry = {
                "file_name": file_name,
                "real_eye_side": real_label,
                "pred_eye_side": "uncertain",
                "left_intensity": None,
                "right_intensity": None,
                "cup_bbox": None,
                "disc_bbox": None,
            }

            # get the bounding boxes
            cup_props = largest_region(cup_mask)
            disc_props = largest_region(disc_mask)

            # return plain entry if the cup or disc is not found
            if not cup_props or not disc_props:
                result.append(entry)
                idx_across_batches += 1
                continue

            # get the bounding boxes
            cup_bbox = cup_props.bbox
            disc_bbox = disc_props.bbox
            entry.update({
                "cup_bbox": cup_bbox,
                "disc_bbox": disc_bbox,
            })

            # crop the disc area
            disc_crop = img[disc_bbox[0]:disc_bbox[2], disc_bbox[1]:disc_bbox[3], :]

            # return plain entry if the disc is not found
            if disc_crop.size == 0:
                result.append(entry)
                idx_across_batches += 1
                continue

            # split the disc into left and right
            _, w, _ = disc_crop.shape # return the height, width, and channel of the disc
            mid = w // 2
            # get only the green channel
            left_green = disc_crop[:, :mid, 1]
            right_green = disc_crop[:, mid:, 1]

            # get the intensities
            left_intensity = float(left_green.sum())
            right_intensity = float(right_green.sum())

            # get the eye side by comparing the intensities
            if left_intensity > right_intensity:
                eye_side = "right"
            elif left_intensity < right_intensity:
                eye_side = "left"
            else:
                eye_side = "uncertain"

            # collect the result
            entry.update({
                "pred_eye_side": eye_side,
                "left_intensity": left_intensity,
                "right_intensity": right_intensity,
            })
            # append the result
            result.append(entry)
            idx_across_batches += 1
    # return the result
    return result

def get_img(path:str, img_size:int) -> tf.Tensor:
    """Import the image and preprocess it

    Args:
        path (str): The path to the image
        img_size (int): The size of the image in square pixels

    Returns:
        tf.Tensor: The preprocessed image
    """
    img = tf.io.read_file(path)                                             # read the image
    img = tf.image.decode_jpeg(img, channels=3)                             # read the jpg file to readable type
    img = tf.image.resize(img, (img_size, img_size), method='nearest')      # change the image size
    img = tf.cast(img, tf.float32) / 255.                                   # normalize the image
    return tf.expand_dims(img, axis=0)                                      # change the shape of the image to simulate batch dataset

def visualize_result_eye_side(model:tf.keras.Model,
                                img_path:str, mask_path:str,
                                threshold:float = .5, img_size:int=128,
                                visualize:bool=True) -> dict:
    """Visualize the result of the eyeside prediction

    Args:
        model (tf.keras.Model): The model to use for prediction
        img_path (str): The path to the image
        mask_path (str): The path to the mask
        threshold (float, optional): The threshold to use for prediction. Defaults to .5.
        img_size (int, optional): The size of the image in square pixels. Defaults to 128.
        visualize (bool, optional): Whether to visualize the result. Defaults to True.

    Returns:
        dict: The result of the eyeside prediction
    """
    # preprocess and get the image
    img = get_img(img_path, img_size)
    # predict the image
    pred_mask = model.predict(img, verbose=0)

    # split the mask image to a binary mask image
    cup_mask = (pred_mask[..., 1] > threshold).astype("uint8")[0]
    disc_mask = (pred_mask[..., 2] > threshold).astype("uint8")[0]

    # create a bounding box for each binary image (will get the coordinate)
    cup_prop = largest_region(cup_mask)
    disc_prop = largest_region(disc_mask)

    # return plain result if the cup or disc is not found
    if not cup_prop or not disc_prop:
        return {
            "file_name": img_path.split("\\")[-1],
            "pred_eye_side": "uncertain",
            "left_intensity": None,
            "right_intensity": None,
            "cup_bbox": None,
            "disc_bbox": None,
        }
    
    # get the bounding boxes
    cup_bbox = cup_prop.bbox
    disc_bbox = disc_prop.bbox

    # crop the disc area
    disc_crop = img[0, disc_bbox[0]:disc_bbox[2], disc_bbox[1]:disc_bbox[3], :].numpy()
    h, w, _ = disc_crop.shape
    mid = w // 2

    # split the disc into left and right
    left_intensity = disc_crop[:, :mid, 1].sum()
    right_intensity = disc_crop[:, mid:, 1].sum()

    # get the eye side by comparing the intensities
    if left_intensity > right_intensity:
        eye_side = "right"
    elif left_intensity < right_intensity:
        eye_side = "left"
    else:
        eye_side = "uncertain"
    
    # visualize the result
    if visualize:
        # create the figure
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        # plot the original mask
        axes[0, 0].imshow(plt.imread(mask_path), cmap="gray")
        axes[0, 0].set_title("Original Mask")
        axes[0, 0].axis("off")
        # plot the predicted mask
        axes[0, 1].imshow(tf.argmax(pred_mask, axis=-1)[0], cmap="gray")
        # axes[0, 1].add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),
        #                                     cup_bbox[3]-cup_bbox[1],
        #                                     cup_bbox[2]-cup_bbox[0],
        #                                     edgecolor='b', facecolor='none'))
        # axes[0, 1].add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),
        #                                     disc_bbox[3]-disc_bbox[1],
        #                                     disc_bbox[2]-disc_bbox[0],
        #                                     edgecolor='c', facecolor='none'))
        axes[0, 1].set_title("Predicted Mask")
        axes[0, 1].axis("off")
        # plot the original image with the bounding boxes
        axes[1, 0].imshow(img[0])
        axes[1, 0].add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),
                                            cup_bbox[3]-cup_bbox[1],
                                            cup_bbox[2]-cup_bbox[0],
                                            edgecolor='b', facecolor='none'))
        axes[1, 0].add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),
                                            disc_bbox[3]-disc_bbox[1],
                                            disc_bbox[2]-disc_bbox[0],
                                            edgecolor='c', facecolor='none'))
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")
        # plot the cropped disc area
        axes[1, 1].imshow(disc_crop)
        axes[1, 1].axvline(x=mid, color="g", linestyle="--", linewidth=2)
        axes[1, 1].set_title(f"Cropped Disc Area ({eye_side} eye)")
        axes[1, 1].axis("off")
        # plot the left green channel
        axes[2, 0].imshow(disc_crop[:, :mid, 1], cmap="gray")
        axes[2, 0].set_title(f"Left Green Channel: {left_intensity:.2f}")
        axes[2, 0].axis("off")
        # plot the right green channel
        axes[2, 1].imshow(disc_crop[:, mid:, 1], cmap="gray")
        axes[2, 1].set_title(f"Right Green Channel: {right_intensity:.2f}")
        axes[2, 1].axis("off")
        # show the figure with tight layout
        plt.tight_layout()
        plt.show()

    # return the result
    return {
        "file_name": img_path.split("\\")[-1],
        "pred_eye_side": eye_side,
        "left_intensity": float(left_intensity),
        "right_intensity": float(right_intensity),
        "cup_bbox": cup_bbox,
        "disc_bbox": disc_bbox,
    }