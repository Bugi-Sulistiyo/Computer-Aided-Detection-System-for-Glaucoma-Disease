import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Accuracy, SparseCategoricalAccuracy, MeanIoU
from tensorflow_addons.metrics import F1Score

tf.keras.backend.clear_session()

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

def remap_mask(mask):
    mask = tf.where(mask == 64, 1, mask)
    mask = tf.where(mask ==255, 2, mask)
    return mask

def load_image(img_path, mask_path, img_size:int=128):
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
    img = tf.image.resize(img, (img_size, img_size), method='nearest')
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (img_size, img_size), method='nearest')
    mask = tf.cast(mask, tf.int32)
    mask = remap_mask(mask)

    return img, mask

def create_dataset(img_paths:list, mask_paths:list, batch_size:int=16):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def custom_unet(input_shape:tuple=(128, 128, 3), num_classes:int=3, filters:list=[16, 32, 64]):
    input_layer = layers.Input(shape=input_shape)
    x = input_layer
    skips = []

    # Encoder
    for filter in filters[:-1]:
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        skips.append(x)
        x = layers.MaxPool2D((2,2))(x)
    
    # Bottleneck
    x = layers.Conv2D(filters[-1], (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters[-1], (3,3), padding='same', activation='relu')(x)

    # Decoder
    for filter, skip in zip(reversed(filters[:-1]), reversed(skips)):
        x = layers.UpSampling2D((2,2))(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
    
    # Output
    output_layer = layers.Conv2D(num_classes, (1,1), activation='softmax')(x)

    return models.Model(input_layer, output_layer)

def train_model(model:tf.keras.Model, trainset, valset, file_name:str, epochs:int=10):
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=[
                            Accuracy(name='acc'),
                            # SparseCategoricalAccuracy(name='px_acc'),
                            # MeanIoU(num_classes=3, name='iou'),
                            # F1Score(num_classes=3, name='f1')
                            ]
                            )
    model.fit(trainset, validation_data=valset, epochs=epochs, verbose=0)
    model.save(f"./../../data/model/{file_name}.h5")
    return model

def get_bounding_box(mask:np.array):
    """get the bounding box of the mask

    Args:
        mask (np.array): the mask of the image

    Returns:
        tuple: the bounding box of the mask and the size of the mask (ymin, ymax, xmin, xmax, height, width)
    """

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    height = ymax - ymin + 1
    width = xmax - xmin + 1

    return ymin, ymax, xmin, xmax, height, width