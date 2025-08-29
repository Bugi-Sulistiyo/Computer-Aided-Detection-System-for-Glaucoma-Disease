import os
import matplotlib.pyplot as plt
from typing import Literal
from tensorflow.keras.utils import plot_model
from .modeling_utils import custom_load_model

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