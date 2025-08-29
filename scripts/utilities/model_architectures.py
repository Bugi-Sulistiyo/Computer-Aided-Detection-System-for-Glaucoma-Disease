from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S

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