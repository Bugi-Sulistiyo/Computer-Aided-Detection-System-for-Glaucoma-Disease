import os
import tensorflow as tf

def load_img_mask(dir_path:str):
    """load images and masks from a directory (train/val/test)

    Args:
        dir_path (str): a directory path where images and masks are stored

    Returns:
        list: a list of fundus images and masks path
    """
    path_imgs = []
    path_masks = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg"):
            path_imgs.append(os.path.join(dir_path, filename))
        elif filename.endswith("_mask.png"):
            path_masks.append(os.path.join(dir_path, filename))
    return path_imgs, path_masks

def load_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (256, 256))
    mask = tf.cast(mask, tf.uint8)

    return img, mask

def create_dataset(img_paths, mask_paths, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset