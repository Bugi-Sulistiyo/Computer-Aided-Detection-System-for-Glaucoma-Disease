# Import the needed package
## package for handling file and directory
import os
## package for handling the image and mask
import numpy as np
## package for visualize the image and mask
import matplotlib.pyplot as plt
## package for modelling
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import load_model

tf.keras.backend.clear_session()

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

    return img, mask

def create_dataset(img_paths:list, mask_paths:list, batch_size:int=16):
    """create a tf.data.Dataset from image and mask paths

    Args:
        img_paths (list): a list of image paths
        mask_paths (list): a list of mask paths
        batch_size (int, optional): the size of batches. Defaults to 16.

    Returns:
        tf.data.Dataset: the batched dataset
    """
    # create a dataset from the image and mask paths
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    # standardize the image and mask
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # shuffle the dataset into a random order and make it a batch
    dataset = dataset.batch(batch_size)
    return dataset

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
    input_layer = layers.Input(shape=input_shape)
    x = input_layer
    skips = []

    # Encoder 
    for filter in filters[:-1]:
        # Extract the image features
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        # store the skip connection
        skips.append(x) 
        # decrese the image size
        x = layers.MaxPool2D((2,2))(x)

    # Bottleneck
    x = layers.Conv2D(filters[-1], (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters[-1], (3,3), padding='same', activation='relu')(x)

    # Decoder
    for filter, skip in zip(reversed(filters[:-1]), reversed(skips)):
        # restore the image size
        x = layers.UpSampling2D((2,2))(x)
        # implement the skip connection
        x = layers.Concatenate()([x, skip])
        # Extract the image features
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filter, (3,3), padding='same', activation='relu')(x)

    # Output
    output_layer = layers.Conv2D(num_classes, (1,1), activation='softmax')(x)
    return models.Model(input_layer, output_layer)

def train_model(model:tf.keras.Model,
                trainset:tf.data.Dataset, valset:tf.data.Dataset, testset:tf.data.Dataset,
                file_name:str, epochs:int=10):
    """train the model and save it

    Args:
        model (tf.keras.Model): the model to be trained
        trainset (tf.data.Dataset): the dataset used for training
        valset (tf.data.Dataset): the dataset used for validation
        testset (tf.data.Dataset): the dataset used for testing
        file_name (str): the name of model to be saved as file
        epochs (int, optional): the number of iteration the training would be done. Defaults to 10.

    Returns:
        tf.keras.Model, str, str: model after trained, the loss value, the accuracy value
    """
    # set the configuration of the model on training
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[SparseCategoricalAccuracy()])
    # train the model
    model.fit(trainset, validation_data=valset, epochs=epochs, verbose=0)
    # save the model into .h5 file
    model.save(f"./../../../data/model/{file_name}.h5")
    #  test the model with testset and getting the loss and accuracy values
    loss, acc = model.evaluate(testset, verbose=0)
    return model, loss, acc

def predict_model(testset:tf.data.Dataset, file_name:str="unet_custom",
                batches:int=1, get_one:bool=True, bucket_choosed:int=0):
    """create the predicted mask by the model

    Args:
        testset (tf.data.Dataset): the dataset containing the image to predict the mask
        file_name (str, optional): the model name saved as .h5 file. Defaults to "unet_custom".
        batches (int, optional): the number of batches want to used. Defaults to 1.
        bucket_choosed (int, optional): the index of batches wanted. Defaults to 0.
        get_one (bool, optional): get one mask only. Defaults to True.

    Returns:
        list=[tf.Tensor], tf.keras.Model: the predicted mask from the model, the model
    """
    pred_mask = []
    # load the saved model
    model = load_model(f"./../../../data/model/{file_name}.h5")
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
        for index in range(img_shown):
            # show the true image
            plt.subplot(3, img_shown, index+1)
            plt.imshow(image[index])
            plt.axis("off")

            # show the true mask
            plt.subplot(3, img_shown, index+(1+img_shown))
            plt.imshow(mask[index], cmap="gray")
            plt.axis("off")

            # show the predicted mask
            plt.subplot(3, img_shown, index+(1+img_shown*2))
            plt.imshow(pred[index], cmap="gray")
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

def calculate_area_CDR(cup_mask:np.array, disc_mask:np.array, bcup_mask:np.array, bdisc_mask:np.array):
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