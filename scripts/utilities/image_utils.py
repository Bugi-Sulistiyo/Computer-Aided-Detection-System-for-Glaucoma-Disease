from tf_clahe import clahe
import tensorflow as tf
import os

from .data_utils import get_file

def augment_clahe(image:tf.Tensor, clip_limit:float):
    """augment image using CLAHE

    Args:
        image (tf.Tensor): the image to be augmented
        clip_limit (float): the clip limit of the CLAHE

    Returns:
        tf.Tensor: the augmented image
    """
    return clahe(image, clip_limit=clip_limit)

def create_aug_img(imgs_path:list, src_dir:str, dst_dir:str, clip_limit:float):
    """create augmented images using CLAHE

    Args:
        imgs_path (list): a list of images to be augmented
        src_dir (str): the source directory where the images are located
        dst_dir (str): the destination directory where the augmented images will be saved
        clip_limit (float): the clip limit of the CLAHE
    """
    for img_path in imgs_path:
        # get image
        image = get_file(img_path, src_dir, "image")
        # augment image
        image = augment_clahe(image, clip_limit)
        # turn back the tensor to image
        image = tf.image.encode_jpeg(image)
        # save the augmented image
        tf.io.write_file(
            os.path.join(
                dst_dir,
                f"{img_path.split('.')[0]}_aug.jpg"),
            image)