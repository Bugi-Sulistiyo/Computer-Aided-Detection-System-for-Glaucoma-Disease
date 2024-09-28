import os
import shutil
import tensorflow as tf
from tf_clahe import clahe

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

def augment_clahe(image:tf.Tensor):
    """augment image using CLAHE

    Args:
        image (tf.Tensor): the image to be augmented

    Returns:
        tf.Tensor: the augmented image
    """
    return clahe(image, clip_limit=1.5)

def create_aug_img(imgs_path:list, src_dir:str, dst_dir:str):
    """create augmented images using CLAHE

    Args:
        imgs_path (list): a list of images to be augmented
        src_dir (str): the source directory where the images are located
        dst_dir (str): the destination directory where the augmented images will be saved
    """
    for img_path in imgs_path:
        # get image
        image = get_file(img_path, src_dir, "image")
        # augment image
        image = augment_clahe(image)
        # turn back the tensor to image
        image = tf.image.encode_jpeg(image)
        # save the augmented image
        tf.io.write_file(
            os.path.join(
                dst_dir,
                f"{img_path.split('.')[0]}_aug.jpg"),
            image)