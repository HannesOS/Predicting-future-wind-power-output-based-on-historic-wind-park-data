from src.data_utils import *
from src.model_utils import *
from src.serialization_utils import *
from architectures.FFNN import CustomFFNN
from architectures.LSTM import CustomLSTM
from architectures.autoencoder import CustomAutoencoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from tensorboard.plugins.hparams import api as hp
from datetime import datetime as dt
from keras.callbacks import CSVLogger
import keras_tuner as kt


def build_model(model_architecture, hyperparameters, loss='mse'):
    """
    Creates, compiles and returns a keras Model. 
    The model will learn with the Adam optimizer.
    During training, the Root Mean Squared Error and the Cumulative Absolute Percentage Error are being tracked.

    Args:
        model_architecture (string): Model architecture
        hyperparameters (ndarray): Hyperparameters used in the Model.
        loss (str, optional): Loss function that should be used. Defaults to 'mse'.

    Returns:
        keras.Model: compiled Keras Model
    """

    model = model_dict[model_architecture](*hyperparameters)
    model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=[tf.keras.metrics.RootMeanSquaredError(),
            cumulative_absolute_percentage_error_kerras_metric]
            )
    return model


def stopping_condition(monitor='val_cumulative_absolute_percentage_error_kerras_metric', patience=300):
    """
    To find the best model, the training will stop if the training starts to perform worse based on a metric on the validation data.
    To account for some randomness, training won't stop instantly after it starts performing worse on the validation data
    but rather continues to train for a certain number of epochs (given by the patience parameter) and will eventually rollback to the
    best validation performance if no better alternative has been found and the training terminates.

    Args:
        monitor (str, optional): Metric which is used to measure performance on validation data. Defaults to 'val_cumulative_absolute_percentage_error_kerras_metric'.
        patience (int, optional): Number of epochs where the performance has to be worse than the previous best before initiating a roll-back and stopping training. Defaults to 300.

    Returns:
        keras.Callback: The Keras callback function for the early stopping condition.
    """

    stop = tf.keras.callbacks.EarlyStopping(
                        monitor=monitor,
                        patience=patience,
                        restore_best_weights=True
                        ) 
    return stop


def train_model(x_train, y_train, x_test,
 y_test, model_architecture, hyperparameters, loss='mse'):
    """
    Trains a Keras Model to the given train data. 
    Training will be stopped early if training performance on the validation data decreases for a certain amount of epochs.
    The model will be serialized using the keras function 'model.save()'.
    Train progress will be logged and a summary will be saved locally. 

    Args:
        x_train (ndarray): Train features.
        y_train (ndarray): Train targets.
        x_test (ndarray): Test features.
        y_test (ndarray): Test targets.
        model_architecture (string): model architecture.
        hyperparameters (ndarray): Hyperparameters used in the model.
        loss (str, optional): Loss function used for training. Defaults to 'mse'.

    Returns:
        keras.Model: Trained Model.
    """

    clear_logs(model_architecture)
    csv_logger = CSVLogger(get_train_log_path(model_architecture), append=True, separator=';')

    early_stopping = stopping_condition()
                
    model = build_model(model_architecture, hyperparameters)

    if model_architecture == 'autoencoder':
        early_stopping = stopping_condition('val_loss') 
        #In an autoencoder we also want to use the features as the target
        y_train = x_train
        y_test = x_test

    # When we are using LSTMs or GRUs we want to use a keras TimeSeriesGenerator as train data
    if model_architecture == 'LSTM' or model_architecture == 'GRU':
        history = model.fit(
                x_train,
                epochs=5000, batch_size=128,
                callbacks=[csv_logger, early_stopping], validation_data=x_test
                )
    else:         
        history = model.fit(
                    x_train, y_train,
                    epochs=5000, batch_size=128,
                    callbacks=[csv_logger, early_stopping], validation_split=0.2
                    )

    model.save(get_trained_model_path(model_architecture))

    summary_log(
                model, model_architecture,
                x_test, y_test,
                0, history,
                hyperparameters 
                )
    return model


def hyperparams_search(x_train, y_train, x_test,
 y_test, model_architecture, possible_layer_units,
  possible_layer_amount, dropout_rates, trial_iterations=20):
    """
    Iterates through the given possible hyperparameters and trains models on all parameter combinations.
    The results of this hyperparameter search are then serialized.

    Args:
        x_train (ndarray): Train features.
        y_train (ndarray): Train targets.
        x_test (ndarray): Test features.
        y_test (ndarray): Test targets.
        model_architecture (string): Model architecture
        possible_layer_units (ndarray): Possible amounts of neurons in a layer as an array(could look something like [32, 64, 128])
        possible_layer_amount (ndarray): Possible amounts of layers in the model as an array (could look something like [1, 2, 3])
        dropout_rates (ndarray): Possible dropout rates in the model (could look something like [0, 0.25, 0.5])
        trial_iterations (int, optional): Determines how often the search process should be repeated. Defaults to 20.

    Returns:
        ndarray: training histories of all models
    """

    clear_logs(model_architecture)
    csv_logger = CSVLogger(get_train_log_path(model_architecture), append=True, separator=';')
    histories = []

    for trial_iteration in range(trial_iterations):
        # Iterate through the hyperparameters
        for layer_amount in possible_layer_amount:
            layer_units = np.random.choice(possible_layer_units, layer_amount)

            for dropout_rate in dropout_rates:

                hyperparameters = [layer_units, dropout_rate]

                early_stopping = stopping_condition()
                
                model = build_model(model_architecture, hyperparameters)

                history = model.fit(
                    x_train, y_train,
                     epochs=5000, batch_size=128,
                      callbacks=[csv_logger, early_stopping], validation_split=0.2, verbose=3
                      )
                histories.append(history)

                model.save(get_trained_model_path(model_architecture))

                summary_log(
                            model, model_architecture,
                            x_test, y_test,
                            trial_iteration, history,
                            hyperparameters 
                            )
    return histories



if __name__ == "__main__":
    model_architecture = 'GRU' #Choose between FFNN, LSTM, autoencoder

    drop_cols = [15, 16] #drop interpolated and capacity columns as they shouldn't matter
    x_train, y_train = get_features_and_targets(csv_path=PATH_TRAIN_DATA, excluded_features=drop_cols, normalize=False)
    x_test, y_test = get_features_and_targets(csv_path=PATH_SOLUTION, excluded_features=drop_cols, normalize=False)


    #Hyperparameters
    #possible_layer_units, possible_layer_amount, dropout_rates  = model_hyperparameters_FFNN.values()
    layer_units = [32, 64]
    dropout_rate = 0
    if model_architecture == 'LSTM' or model_architecture == 'GRU':
        x_train = get_LSTM_data_generator(csv_path=PATH_TRAIN_DATA, excluded_features=drop_cols, window_size=10)
        x_test = get_LSTM_data_generator(csv_path=PATH_SOLUTION, excluded_features=drop_cols, window_size=10)

    if model_architecture == 'random_forest':
        regr = RandomForestRegressor()
        #regr.fit(x_train, y_train)
        scores = cross_val_score(regr, x_train, y_train, cv=2)
        print(scores)
        prediction = regr.predict(x_test)
        CAPE = cumulative_absolute_percentage_error(y_test.to_numpy(dtype=float), prediction.flatten())
        print(f'Cumulative Absolute Percentage Error on test data: {CAPE} \n\n\n\n')

    
    #layer_units = [64, 128, 256, 512, 1024]
    #dropout_rate = 0
    #model = train_model(x_train, y_train, x_test, y_test, model_architecture, hyperparameters= [5, len(x_train.columns)])
    #train_model(x_train, y_train, x_test, y_test, 'FFNN', hyperparameters= [layer_units, dropout_rate])
    #autoencoder = tf.keras.models.load_model('trained_models\\best\\autoencoder', custom_objects={"cumulative_absolute_percentage_error_kerras_metric": cumulative_absolute_percentage_error_kerras_metric})
    #encoder = autoencoder.encoder
    #x_train_encoded = encoder.predict(x_train)
    #x_test_encoded = encoder.predict(x_test)


    #model = tf.keras.models.load_model('trained_models\\best\\FFNN', custom_objects={"cumulative_absolute_percentage_error_kerras_metric": cumulative_absolute_percentage_error_kerras_metric})
    #train_model(x_train, y_train, x_test, y_test, model_architecture, hyperparameters= [layer_units, dropout_rate])
    summary_log(model, model_architecture, x_test, y_test, trial, history, hyperparameters)
    #prediction = model.predict(x_test_encoded)

    #hyper_params_search(x_train_encoded, y_train, x_test_encoded,
    #    y_test, model_architecture, possible_layer_units,
    #       possible_layer_amount, dropout_rates)
           
    #CAPE = cumulative_absolute_percentage_error(y_test.to_numpy(dtype=float), prediction.flatten())
    #print(CAPE)





