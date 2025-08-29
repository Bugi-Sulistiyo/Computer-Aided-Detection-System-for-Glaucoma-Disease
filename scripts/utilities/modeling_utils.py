import tensorflow as tf
import os
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from .data_utils import calculate_weight, add_sample_weight
from tensorflow.keras.models import load_model
from typing import Literal

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

def custom_load_model(model_path:str):
    """load the model with custom metric

    Args:
        model_path (str): the complete path of the model

    Returns:
        tf.keras.Model: the model
    """
    return load_model(model_path, custom_objects={"mean_px_acc": mean_px_acc})

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