import numpy as np
import tensorflow as tf
from architectures.FFNN import CustomFFNN
from architectures.LSTM import CustomLSTM
from architectures.GRU import CustomGRU
from architectures.autoencoder import CustomAutoencoder
import tensorflow.keras.backend as K


BASIS = 122400 #installed capacity of wind park in kW

#Model classes for each model architecture
model_dict = {'FFNN' : CustomFFNN, 'LSTM' : CustomLSTM, 'GRU' : CustomGRU, 'autoencoder': CustomAutoencoder}


#Possible hyperparamaters for the different model architectures.
model_hyperparameters_FFNN = {
                            'possible_layer_units' : [32, 64, 128, 256, 512, 1024],
                            'possible_layer_amount' : np.arange(4, 9)[::-1],
                            'dropout_rates' : [0, 0.25]
                            }

model_hyperparameters_LSTM = {
                            'possible_layer_units' : [32, 64, 128, 256, 512, 1024],
                            'possible_layer_amount' : np.arange(1, 5)[::-1],
                            'dropout_rates' : [0, 0.25]
                            }

model_hyperparameters_GRU = {
                            'possible_layer_units' : [32, 64, 128, 256, 512, 1024],
                            'possible_layer_amount' : np.arange(1, 5)[::-1],
                            'dropout_rates' : [0, 0.25]
                            }


def get_hyperparameters(model_architecture):
    if model_architecture == 'FFNN':
        return model_hyperparameters_FFNN
    elif model_architecture == 'LSTM':
        return model_hyperparameters_LSTM
    elif model_architecture == 'GRU':
        return model_hyperparameters_GRU
    else:
        exit('Please specify a valid model architecture')


def cumulative_absolute_percentage_error_kerras_metric(y_true, y_pred):
    """
    This function can be used by Keras during the training of a model to calculate the cumulative absolute percentage error (CAPE) of a prediction.

    Args:
        y_true (ndarray): Actual targets.
        y_pred (ndarray): Predicted targets.

    Returns:
        float: CAPE of the prediction
    """

    return K.abs(y_pred - y_true)/BASIS


def cumulative_absolute_percentage_error(y_true, y_pred):
    """
    This function can be used by the user to calculate the cumulative absolute percentage error (CAPE) of a prediction.

    Args:
        y_true (ndarray): Actual targets.
        y_pred (ndarray): Predicted targets.

    Returns:
        float: CAPE of the prediction
    """

    nominator = np.sum(np.abs(y_true[:len(y_pred)] - y_pred))
    denominator = BASIS * len(y_pred)
    return nominator / denominator