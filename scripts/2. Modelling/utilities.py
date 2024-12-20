# Import the needed package
## package for handling file and directory
import os
import shutil
## package for restric param value
from typing import Literal
## package for handling the image and mask
import numpy as np
## package for handling the dataframe
import pandas as pd
## package for visualize the image and mask
import matplotlib.pyplot as plt
## package for handling the mask bounding box
from skimage.measure import label, regionprops
## package for handling the dataset in general
import tensorflow as tf
## package for image augmentation
from tf_clahe import clahe
## package for modelling
### create the model
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization, Dropout
### compile the model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import Callback, TensorBoard
### predict requirement
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
### visualize model architecture
from tensorflow.keras.utils import plot_model

# Clear the session
tf.keras.backend.clear_session()

# Function for data preparation
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

# Function for image augmentation
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

# function for modeling
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
    dataset = dataset.prefetch(tf.data. AUTOTUNE)
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

def custom_unet(input_shape:tuple=(128, 128, 3), num_classes:int=3, filters:list=[16, 32, 64]):
    """create a custom unet model dynamicly

    Args:
        input_shape (tuple, optional): the image shape inputed to the model. Defaults to (128, 128, 3).
        num_classes (int, optional): number of classes (background, cup, disc). Defaults to 3.
        filters (list, optional): the filter used in the model structure. Defaults to [16, 32, 64].

    Returns:
        tf.keras.Model: the unet model
    """
    # input layer of the model
    input_layer = Input(shape=input_shape)
    x = input_layer
    skips = []

    # Encoder 
    for filter in filters[:-1]:
        # Extract the image features
        x = Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        # store the skip connection
        skips.append(x) 
        # decrese the image size
        x = MaxPool2D((2,2))(x)

    # Bottleneck
    x = Conv2D(filters[-1], (3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters[-1], (3,3), padding='same', activation='relu')(x)

    # Decoder
    for filter, skip in zip(reversed(filters[:-1]), reversed(skips)):
        # restore the image size
        x = UpSampling2D((2,2))(x)
        # implement the skip connection
        x = Concatenate()([x, skip])
        # Extract the image features
        x = Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    # Output
    output_layer = Conv2D(num_classes, (1,1), activation='softmax')(x)

    # Create the model
    return Model(input_layer, output_layer)

def mobilenet_model(input_shape:tuple=(128, 128, 3), num_classes:int=3, filters:list=[16, 32, 64]):
    """create a mobilenet model for semantic segmentation

    Args:
        input_shape (tuple): Shape of the input images. Defaults to (128, 128, 3).
        num_classes (int): Number of output classes. Defaults to 3.
        filters (list): Filters for the decoder layers. Defaults to [16, 32, 64].

    Returns:
        tf.keras.Model: The MobileNet segmentation model.
    """
    # Load the MobileNet model
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights='imagenet')

    # Get the output of the encoder
    x = base_model.output  # Output features from the deepest layer

    # Decoder
    for filter in reversed(filters):
        # Upsample and apply convolution layers
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filter, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
    
    # Upsample the output of the encoder to the input size
    while x.shape[1] < input_shape[0]:
        x =  UpSampling2D((2,2))(x)
    # Output layer for segmentation
    output_layer = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the model
    return Model(inputs=base_model.input, outputs=output_layer)

def efficientnet_model(input_shape:tuple=(128, 128, 3), num_classes:int=3, filters:list=[16, 32, 64]):
    """create a efficientnet model for semantic segmentation

    Args:
        input_shape (tuple): Shape of the input images. Defaults to (128, 128, 3).
        num_classes (int): Number of output classes. Defaults to 3.
        filters (list): Filters for the decoder layers. Defaults to [16, 32, 64].

    Returns:
        tf.keras.Model: The EfficientNet segmentation model.
    """
    # Load the EfficientNet model
    base_model = EfficientNetV2S(input_shape=input_shape, include_top=False, weights='imagenet')

    # Get the skip layers from the base model
    skip_layers = [
        base_model.get_layer("block2a_project_bn").output,  # shallow features
        base_model.get_layer("block3a_project_bn").output,  # mid-level features
        base_model.get_layer("block4a_project_bn").output   # deep features
    ]

    # Get the output of the encoder
    x = base_model.output  # Output features from the deepest layer
    
    # Decoder
    for index, filter in enumerate(reversed(filters)):
        # Upsample and apply convolution layers
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, skip_layers[-(index+1)]])
        x = Conv2D(filter, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    # Upsample the output of the encoder to the input size
    while x.shape[1] < input_shape[0]:
        x = UpSampling2D((2, 2))(x)
    # Output layer for segmentation
    output_layer = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Create the model
    return Model(inputs=base_model.input, outputs=output_layer)

def mean_px_acc(y_true:tf.Tensor, y_pred:tf.Tensor):
    """a custom metric to calculate the mean pixel accuracy

    Args:
        y_true (tf.Tensor): the true mask
        y_pred (tf.Tensor): the predicted mask

    Returns:
        tf.Tensor: the mean pixel accuracy
    """
    # get the index of the maximum value in the mask
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    # get the component of the number of correct pixels and total pixels
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32), axis=[1, 2])
    total_pixels = tf.reduce_sum(tf.ones_like(y_true, dtype=tf.float32), axis=[1,2])
    # calculate the mean pixel accuracy
    return tf.reduce_mean(correct_pixels / total_pixels)

class AUCStoppingCallback(Callback):
    """a custom callback to stop the training when the AUC value reaches the target value

    Args:
        Callback (tf.keras.callbacks.Callback): the callback class from tensorflow
    """
    def __init__(self, target_auc:float=.98):
        super(AUCStoppingCallback, self).__init__()
        self.target_auc = target_auc
    
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_auc'] is not None and logs['val_auc'] >= self.target_auc:
            print(f"\nReached {self.target_auc} AUC value. Stopping the training on epoch {epoch}")
            self.model.stop_training = True

def train_model(model:tf.keras.Model,
                trainset:tf.data.Dataset, valset:tf.data.Dataset, testset:tf.data.Dataset,
                model_path:str, file_name:str, tensorboard_dir:str, epochs:int=10):
    """train the model and save it

    Args:
        model (tf.keras.Model): the model to be trained
        trainset (tf.data.Dataset): the dataset used for training
        valset (tf.data.Dataset): the dataset used for validation
        testset (tf.data.Dataset): the dataset used for testing
        model_path (str): the path where the model will be saved
        file_name (str): the name of model to be saved as file
        tensorboard_dir (str): the directory where the tensorboard log will be saved
        epochs (int, optional): the number of iteration the training would be done. Defaults to 10.

    Returns:
        tf.keras.Model, str, str: model after trained, the loss value, the accuracy value
    """
    # get the weight of each label in the mask images
    weights = calculate_weight(trainset)

    # set the configuration of the model on training
    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(from_logits=False),
        weighted_metrics=[mean_px_acc,
                        AUC(name="auc"),
                        Precision(name="precision"),
                        Recall(name="recall")])
    # train the model
    history = model.fit(
        trainset.map(lambda x, y: add_sample_weight(x, y, weights)),
        validation_data=valset.map(lambda x, y: add_sample_weight(x, y, weights)),
        epochs=epochs, callbacks=[AUCStoppingCallback(target_auc=.98),
                                    TensorBoard(log_dir=tensorboard_dir)],
        verbose=1)
    # save the model into .h5 file
    model.save(os.path.join(model_path, f"{file_name}.h5"))
    #  test the model with testset and getting the loss and accuracy values
    return model, history, model.evaluate(testset, verbose=0)

# function for evaluation
def custom_load_model(model_path:str):
    """load the model with custom metric

    Args:
        model_path (str): the complete path of the model

    Returns:
        tf.keras.Model: the model
    """
    return load_model(model_path, custom_objects={"mean_px_acc": mean_px_acc})

def visualize_model_architec(models_name:list, model_path:str, flow_dir:Literal["RL", "TB"]="TB",
                            show_layer_names:bool=False, show_dtype:bool=False, show_shapes:bool=False):
    """visuazlie model architectur. could only be run if graphviz is installed

    Args:
        models_name (list): a list of model name
        model_path (str): the path where trained model is storeed
        flow_dir (Literal["RL", "TB"], optional): the direction of plot. Defaults to "TB".
        show_layer_names (bool, optional): include layer name or not. Defaults to False.
        show_dtype (bool, optional): include layer type or not. Defaults to False.
        show_shapes (bool, optional): include layer shape or not. Defaults to False.

    Returns:
        list: all image path that show model architecture
    """
    # initiate empty dict to store model
    models = {}
    # import all the model
    for name in models_name:
        models[name] = custom_load_model(os.path.join(model_path, f"{name}.h5"))
    
    # visualize the model architecture
    for file_name, model in models.items():
        plot_model(
            model,
            to_file=os.path.join(model_path, f"architec_{file_name}.png"),
            dpi=300,
            rankdir=flow_dir,
            show_dtype=show_dtype,
            show_layer_names=show_layer_names,
            show_shapes=show_shapes
        )
    
    # return the image path
    return [os.path.join(model_path, f"architec_{file_path}.png") for file_path in models_name]

def predict_model(testset:tf.data.Dataset, model_path:str,
                file_name:Literal["efnet_model_aug", "efnet_model_ori",
                                    "mnet_model_aug", "mnet_model_ori",
                                    "unet_model_aug", "unet_model_ori"]="unet_model_ori",
                batches:int=1, get_one:bool=True, bucket_choosed:int=0):
    """create the predicted mask by the model

    Args:
        testset (tf.data.Dataset): the dataset containing the image to predict the mask
        model_path (str): the path where the model is saved
        file_name (Literal["efnet_model_aug", "efnet_model_ori",
                            "mnet_model_aug", "mnet_model_ori",
                            "unet_model_aug", "unet_model_ori], optional): the name of the model want to be used. Defaults to "unet_model_ori"
        batches (int, optional): the number of batches want to used. Defaults to 1.
        bucket_choosed (int, optional): the index of batches wanted. Defaults to 0.
        get_one (bool, optional): get one mask only. Defaults to True.

    Returns:
        list=[tf.Tensor], tf.keras.Model: the predicted mask from the model, the model
    """
    pred_mask = []
    # load the saved model
    model = custom_load_model(os.path.join(model_path, f"{file_name}.h5"))
    # extract the image from the dataset
    for bucket_num, (images, _) in enumerate(testset.take(batches)):
        if bucket_num == bucket_choosed and get_one:
            pred_mask = model.predict(images)
            break
        # predict the masks from the given images
        pred_mask.append(model.predict(images))
    return pred_mask, model

def split_disc_cup_mask(pred_mask, treshold:float=0.1, img_idx:int=13):
    """split the disc and cup section from the predicted mask

    Args:
        pred_mask (tf.Tensor): the predicted mask from model
        treshold (float, optional): the treshold used to make mask as binary. Defaults to 0.1.
        img_idx (int, optional): the index of mask that want to be visualized. Defaults to 13.

    Returns:
        tf.Tensor: the result of the splitted mask based on the label
    """
    # devide the mask into two separate mask
    cup_mask = pred_mask[..., 1]
    disc_mask = pred_mask[..., 2]

    # transform the mask image into a binary mask image
    binary_cup_mask = tf.where(cup_mask > treshold, 1, 0)
    binary_disc_mask = tf.where(disc_mask > treshold, 1, 0)

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

def visualize_pred_mask(testset:tf.data.Dataset, model:tf.keras.Model, img_shown:int=4):
    """visualize the predicted mask from the model result

    Args:
        testset (tf.data.Dataset): the dataset used to test the model
        model (tf.keras.Model): the model to be tested
        img_shown (int, optional): number of image to be shown. Defaults to 4.
    """
    for image, mask in testset.take(1):
        # infer the mask from the model
        pred = model.predict(image, verbose=0)
        plt.figure(figsize=(15, 15))
        for index in range(img_shown):
            # show the true image
            plt.subplot(3, img_shown, index+1)
            plt.imshow(image[index])
            plt.title("input")
            plt.axis("off")

            # show the true mask
            plt.subplot(3, img_shown, index+(1+img_shown))
            plt.imshow(tf.argmax(mask[index], axis=-1), cmap="jet")
            plt.title("label")
            plt.axis("off")

            # show the predicted mask
            plt.subplot(3, img_shown, index+(1+img_shown*2))
            plt.imshow(pred[index], cmap="jet")
            plt.title("prediction")
            plt.axis("off")
        break

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

def visualize_bounding_box(label:str, mask:np.array, bmask:np.array, ymin:int, ymax:int, xmin:int, xmax:int):
    """visualize the bounding box of the mask

    Args:
        label (str): label name of the mask
        mask (np.array): the predicted mask
        bmask (np.array): the binary mask
        ymin (int): the minimum y value of the mask
        ymax (int): the maximum y value of the mask
        xmin (int): the minimum x value of the mask
        xmax (int): the maximum x value of the mask
    """
    # show the bounding box of the mask
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray")
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='r', facecolor='none'))
    plt.text(xmin, ymin, label, color='r')
    plt.title("Original Mask")
    plt.axis("off")

    # show the binary mask
    plt.subplot(1, 2, 2)
    plt.imshow(bmask, cmap="gray")
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor='r', facecolor='none'))
    plt.text(xmin, ymin, label, color='r')
    plt.title("Binary Mask")
    plt.axis("off")
    plt.show()

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