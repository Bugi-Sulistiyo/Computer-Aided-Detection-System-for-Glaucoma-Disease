import os
import tensorflow as tf
from tensorflow.keras import layers, models

tf.keras.backend.clear_session()

def load_data(img_dir:str, mask_dir:str, img_size:int=128, batch_size:int=32):
    """load images and masks from a directory

    Args:
        img_dir (str): the directory path where images are stored
        mask_dir (str): the directory path where masks are stored
        img_size (int, optional): image resolution in 1:1. Defaults to 128.
        batch_size (int, optional): the size of dataset batches. Defaults to 32.

    Returns:
        tf.data.Dataset: a dataset of images and masks
    """
    return tf.data.Dataset.zip((
        tf.keras.utils.image_dataset_from_directory(img_dir,
                                                    label_mode=None,
                                                    image_size=(img_size, img_size),
                                                    batch_size=batch_size),
        tf.keras.utils.image_dataset_from_directory(mask_dir,
                                                    label_mode=None,
                                                    image_size=(img_size, img_size),
                                                    batch_size=batch_size)
    ))

def augment(img:tf.Tensor, mask:tf.Tensor, seed:int=55):
    """augment image and mask

    Args:
        img (tf.Tensor): image
        mask (tf.Tensor): mask
        seed (int, optional): random seed. Defaults to 55.

    Returns:
        tf.Tensor: augmented image and mask
    """
    img_seed = tf.random.experimental.Generator.from_seed(seed)

    img = tf.image.stateless_random_flip_left_right(img, seed=img_seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=img_seed)
    img = tf.image.stateless_random_flip_up_down(img, seed=img_seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=img_seed)
    # img = tf.image.random_brightness(img, 0.2, seed=img_seed)
    # img = tf.image.random_contrast(img, 0.2, 0.5, seed=img_seed)
    return img, mask

def normalize(img:tf.Tensor, mask:tf.Tensor):
    """normalize image and mask

    Args:
        img (tf.Tensor): image
        mask (tf.Tensor): mask

    Returns:
        tf.Tensor: normalized image and mask
    """
    img = tf.cast(img, tf.float32) / 255.0
    # mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask, tf.int32)
    return img, mask

def prepare_data(img_dir:str, mask_dir:str, img_size:int=128, batch_size:int=32, seed:int=55):
    """prepare data for training

    Args:
        img_dir (str): the directory path where images are stored
        mask_dir (str): the directory path where masks are stored
        img_size (int, optional): image resolution in 1:1. Defaults to 128.
        batch_size (int, optional): the size of dataset batches. Defaults to 32.
        seed (int, optional): random seed. Defaults to 55.

    Returns:
        tf.data.Dataset: a dataset of images and masks
    """
    dataset = load_data(img_dir, mask_dir, img_size, batch_size)
    # dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)

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

def load_image(img_path:str, mask_path:str, img_size:int=128):
    """load and preprocess image and mask

    Args:
        img_path (str): the path of the image
        mask_path (str): the path of the mask
        img_size (int, optional): the resolution of img 1:1. Defaults to 128.

    Returns:
        tf.Tensor: image and mask
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (img_size, img_size))
    mask = tf.cast(mask, tf.uint8)

    return img, mask

def create_dataset(img_paths:list, mask_paths:list, batch_size:int=32):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def custom_unet(input_shape:tuple=(128, 128, 3), num_classes:int=3, filters:list=[16, 32, 64]):
    inputs = layers.Input(shape=input_shape)

    # Encoder: Down-sampling
    cv1 = layers.Conv2D(filters[0], (3,3), padding='same', activation='relu')(inputs)
    cv1 = layers.Conv2D(filters[0], (3,3), padding='same', activation='relu')(cv1)
    mp1 = layers.MaxPool2D((2,2))(cv1)

    cv2 = layers.Conv2D(filters[1], (3,3), padding='same', activation='relu')(mp1)
    cv2 = layers.Conv2D(filters[1], (3,3), padding='same', activation='relu')(cv2)
    mp2 = layers.MaxPool2D((2,2))(cv2)

    cv3 = layers.Conv2D(filters[2], (3,3), padding='same', activation='relu')(mp2)
    cv3 = layers.Conv2D(filters[2], (3,3), padding='same', activation='relu')(cv3)

    # Decoder: Up-sampling
    up1 = layers.UpSampling2D((2,2))(cv3)
    up1 = layers.Concatenate()([up1, cv2])
    cv4 = layers.Conv2D(filters[1], (3,3), padding='same', activation='relu')(up1)
    cv4 = layers.Conv2D(filters[1], (3,3), padding='same', activation='relu')(cv4)

    up2 = layers.UpSampling2D((2,2))(cv4)
    up2 = layers.Concatenate()([up2, cv1])
    cv5 = layers.Conv2D(filters[0], (3,3), padding='same', activation='relu')(up2)
    cv5 = layers.Conv2D(filters[0], (3,3), padding='same', activation='relu')(cv5)

    # x = inputs

    # # Encoder
    # skips = []
    # for filter in filters[:-1]:
    #     x = layers.Conv2D(filter, 3, padding='same', activation='relu')(x)
    #     x = layers.Conv2D(filter, 3, padding='same', activation='relu')(x)
    #     skips.append(x)
    #     x = layers.MaxPool2D()(x)

    # # Bottleneck
    # x = layers.Conv2D(filters[-1], 3, padding='same', activation='relu')(x)
    # x = layers.Conv2D(filters[-1], 3, padding='same', activation='relu')(x)

    # # Decoder
    # filters = sorted(filters[:-1], reverse=True)
    # skips = skips[::-1]
    # for i, filter in enumerate(filters[:-1]):
    #     x = layers.UpSampling2D()(x)
    #     x = layers.Concatenate()([x, skips[i]])
    #     x = layers.Conv2D(filter, 3, padding='same', activation='relu')(x)
    #     x = layers.Conv2D(filter, 3, padding='same', activation='relu')(x)

    # Output
    outputs = layers.Conv2D(num_classes, (1,1), activation='softmax')(cv5)
    return models.Model(inputs, outputs)