########################################################################################################################################################
#                                                                                                                                                      #
#   Using TensorFlow to predict future wind power output based on historic wind park data - Bremen Big Data Challenge 2018                             #
#   Authors: Hannes Erbis, Marty Schüller                                                                                                              #
#   Osnabrück University                                                                                                                               #
#   Implementing ANNs with TensorFlow                                                                                                                  #
#   Winter semester 2021/22                                                                                                                            #
#   Project                                                                                                                                            #
#   This module can be seen as the main program for this project                                                                                       #
#   See https://github.com/HannesOS/Predicting-future-wind-power-output-based-on-historic-wind-park-data and project report for more information       #
#                                                                                                                                                      #
########################################################################################################################################################                                                                                                                                                       

import tensorflow as tf
import numpy as np
from src.data_utils import *
from src.model_utils import *
from src.serialization_utils import *
from train import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from tensorboard.plugins.hparams import api as hp
from datetime import datetime as dt




######################################################################################################################
# Load data

# We want to drop the columns 'Interpolated', 'Available capacity' as it generally yields better results
excluded_features = [15, 16]


#Test data for when we use a Feed-Forward network
x_test, y_test = get_features_and_targets(csv_path=PATH_SOLUTION, model_architecture='FFNN', excluded_features=excluded_features, normalize=False)
#Test data for when we use an LSTM or GRU
timeseries_data_test = get_features_and_targets(csv_path=PATH_SOLUTION, model_architecture='LSTM', excluded_features=excluded_features, normalize=False)


# This autoencoder reduces the dimensionality of the data to 5
custom_objects = {"cumulative_absolute_percentage_error_kerras_metric": cumulative_absolute_percentage_error_kerras_metric}
model_autoencoder = tf.keras.models.load_model('trained_models\\best\\autoencoder', custom_objects=custom_objects)
print(f'Autoencoder reproduces the originial input of the test data with a mean squared error of {model_autoencoder.evaluate(x_test, x_test)[0]}')


x_test_encoded, _ = get_features_and_targets(csv_path=PATH_SOLUTION, model_architecture='FFNN', excluded_features=excluded_features, normalize=False, autoencoder=model_autoencoder)
timeseries_data_test_encoded = get_features_and_targets(csv_path=PATH_SOLUTION, model_architecture='LSTM', excluded_features=excluded_features, normalize=False, autoencoder=model_autoencoder)


# To predict the energy outputs for the different models we define some helper function
# For the challenge, the Cumulative Absolute Percentage Error (CAPE) is the metric that decides how good our prediction is
def CAPE(real, prediction):
    return cumulative_absolute_percentage_error(y_test.to_numpy(dtype=float), prediction.flatten())


# Prints out the results and saves them
def predict_output(model, model_architecture, x_test):
    prediction = model.predict(x_test)
    np.savetxt(f'predictions\\prediction_{model_architecture}.csv', prediction)
    print(f'Energy output prediction: {prediction}')
    print(f'CAPE: {CAPE(x_test, prediction)}')


# Now we load the best trained models we obtained for each architecture. 
# See src.train.py for details on how these models where obtained

# Now we load the acutal predicting models
model_FFNN = tf.keras.models.load_model('trained_models\\best\\FFNN_15features', custom_objects=custom_objects)
model_FFNN_encoded = tf.keras.models.load_model('trained_models\\best\\FFMM_15features_encoded_test_data_better', custom_objects=custom_objects)

model_LSTM = tf.keras.models.load_model('trained_models\\best\\LSTM', custom_objects=custom_objects)
model_LSTM_encoded = tf.keras.models.load_model('trained_models\\best\\LSTM_encoded', custom_objects=custom_objects)

model_GRU = tf.keras.models.load_model('trained_models\\best\\GRU', custom_objects=custom_objects)
model_GRU_encoded = tf.keras.models.load_model('trained_models\\best\\GRU_encoded', custom_objects=custom_objects)


# We now predict the energy outputs given our different models

print("Predicting the energy output using an FFNN model:")
model_FFNN.summary()
predict_output(model_FFNN, 'FFNN', x_test)

print("\n\n\nPredicting the energy output using an FFNN model with autoencoded data:")
model_FFNN_encoded.summary()
predict_output(model_FFNN_encoded, 'FFNN_encoded', x_test_encoded)


print("\n\n\nPredicting the energy output using an LSTM model:")
model_LSTM.summary()
predict_output(model_LSTM, 'LSTM', timeseries_data_test)

print("\n\n\nPredicting the energy output using an LSTM model with autoencoded data:")
model_LSTM_encoded.summary()
predict_output(model_LSTM_encoded, 'LSTM_encoded', timeseries_data_test_encoded)


print("\n\n\nPredicting the energy output using a GRU model:")
model_GRU.summary()
predict_output(model_GRU, 'GRU', timeseries_data_test)

print("\n\n\nPredicting the energy output using a GRU model with autoencoded data:")
model_GRU_encoded.summary()
predict_output(model_GRU_encoded, 'GRU_encoded', timeseries_data_test_encoded)


print("The best model we tried is therefore a Feed-Forward-Network with autoencoded data")




