## package for handling file and directory
import os
import shutil
## package for handling the dataset in general
import tensorflow as tf
## package for handling the image and mask
import numpy as np

def copy_images(files:list, subset:str, src_dir:str, dst_dir:str):
    """copy images from source directory to destination directory

    Args:
        files (list): a list of images to be copied
        subset (str): the subset of the images
        src_dir (str): the source directory where the images are located
        dst_dir (str): the destination directory where the images will be copied

    Returns:
        str: the status of the copying process
    """
    for file in files:
        try:
            shutil.copy(os.path.join(src_dir, file),
                        os.path.join(dst_dir, subset, file))
        except FileNotFoundError:
            return f'{file} not found'
    return f'{subset} done'

def get_file(file_path:str, src_path:str, file_type:str):
    """import image or mask file

    Args:
        file_path (str): the file path of the image or mask
        src_path (str): the source directory where the image or mask is located
        file_type (str): the type of file to be imported (image or mask)

    Returns:
        tf.Tensor: the image or mask file
    """
    try:
        # read file
        file = tf.io.read_file(os.path.join(src_path, file_path))
        # decode file
        if file_type == 'image':
            file = tf.image.decode_jpeg(file, channels=3)
        elif file_type == 'mask':
            file = tf.image.decode_png(file, channels=1)
        # resize file
        file = tf.image.resize(file, (512, 512), method="nearest")
        return file
    except FileNotFoundError:
        return f'{file_path} not found'

def load_img_mask(dir_path:str):
    """load images and masks from a directory (train/val/test)

    Args:
        dir_path (str): a directory path where images and masks are stored

    Returns:
        list: a list of fundus images and masks path
    """
    # an empty list to store the path of images and masks
    path_imgs = []
    path_masks = []

    # iterate over the files in the directory
    for filename in os.listdir(dir_path):
        # getting the image that ends with .jpg
        if filename.endswith(".jpg"):
            path_imgs.append(os.path.join(dir_path, filename))
        # getting the mask that ends with _mask.png
        elif filename.endswith("_mask.png"):
            path_masks.append(os.path.join(dir_path, filename))
    return path_imgs, path_masks

def remap_mask(mask:tf.Tensor):
    """remap the mask values to 0, 1, 2 (background, cup, disc)

    Args:
        mask (tf.Tensor): the mask of the image

    Returns:
        tf.Tensor: remapped mask
    """
    mask = tf.where(mask == 64, 1, mask) # change the cup value into 1
    mask = tf.where(mask ==255, 2, mask) # change the disc value into 2
    return mask

def load_image(img_path:str, mask_path:str, img_size:int=128):
    """load and preprocess image and mask

    Args:
        img_path (str): the path of the image
        mask_path (str): the path of the mask
        img_size (int, optional): the resolution of img 1:1. Defaults to 128.

    Returns:
        tf.Tensor: image and mask
    """
    # import and standardize the image
    img = tf.io.read_file(img_path)                                     # read the image file
    img = tf.image.decode_jpeg(img, channels=3)                         # decode the image into 3 channels
    img = tf.image.resize(img, (img_size, img_size), method='nearest')  # resize the image into the desired size
    img = tf.cast(img, tf.float32) / 255.                               # change the image value into float32 and normalize it

    # import and standardize the mask
    mask = tf.io.read_file(mask_path)                                       # read the mask file
    mask = tf.image.decode_png(mask, channels=1)                            # decode the mask into 1 channel
    mask = tf.image.resize(mask, (img_size, img_size), method='nearest')    # resize the mask into the desired size
    mask = tf.cast(mask, tf.int32)                                          # change the mask value into an integer
    # change the mask value into an integer
    mask = remap_mask(mask)
    # convert to one-hot encoding
    mask = tf.one_hot(tf.squeeze(mask), depth=3)
    mask = tf.cast(mask, tf.int32)

    return img, mask

def create_dataset(img_paths:list, mask_paths:list, img_size:int=128, batch_size:int=16):
    """create a tf.data.Dataset from image and mask paths

    Args:
        img_paths (list): a list of image paths
        mask_paths (list): a list of mask paths
        img_size (int, optional): the resolution of img 1:1. Defaults to 128.
        batch_size (int, optional): the size of batches. Defaults to 16.

    Returns:
        tf.data.Dataset: the batched dataset
    """
    # create a dataset from the image and mask paths
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    # standardize the image and mask
    dataset = dataset.map(lambda x, y: load_image(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    # shuffle the dataset into a random order
    dataset = dataset.shuffle(512)
    # shuffle the dataset into a random order and make it a batch
    dataset = dataset.batch(batch_size)
    # prefetch the dataset to make it faster
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def calculate_weight(dataset:tf.data.Dataset, num_classes:int=3):
    """calculate the weight of each label in the mask images

    Args:
        dataset (tf.data.Dataset): the dataset containing the image and mask (the batched dataset)
        num_classes (int, optional): the number count of existing class. Defaults to 3.

    Returns:
        dict: a dictionary containing the average weight of each label
    """
    # an empty dictionary to store the weight of each label
    weights = {}
    # populate the keys of the dictionary with the label and define the list of weights
    for label in range(num_classes):
        weights[label] = []
    # iterate over the dataset to calculate the weight of each label on each mask
    for _, masks in dataset:
        for mask in masks:
            count_px = {}
            # extract the number of pixel for each label
            for i in range(num_classes):
                count_px[i] = np.sum(mask[..., i])
            # calculate the weight of each label on a single mask
            for i in range(num_classes):
                weights[i].append((1 / count_px[i])
                                    * (np.sum([count_px[j] for j in range(num_classes)]) / num_classes))
    # calculate the average weight of each label
    for label, pxs in weights.items():
        weights[label] = round(np.mean(pxs), 4)
    return weights

def add_sample_weight(img:tf.Tensor, mask:tf.Tensor, weights:dict):
    """create a sample weight for each mask image

    Args:
        img (tf.Tensor): the image inside the bathced dataset
        mask (tf.Tensor): the mask inside the bathced dataset
        weights (dict): the weight of each label in the mask images

    Returns:
        tf.data.Dataset: the image, mask, and sample weight in the bathced dataset
    """
    # recalculate the weight of each label with constraint that the sum of the weight is 1
    class_weights = tf.constant(list(weights.values()))
    class_weights = class_weights / tf.reduce_sum(class_weights)
    # create an image of sample weight
    sample_weights = tf.reduce_sum(class_weights * tf.cast(mask, tf.float32), axis=-1)
    return img, mask, sample_weights