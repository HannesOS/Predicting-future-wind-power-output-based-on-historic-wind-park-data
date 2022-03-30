from datetime import datetime as dt
from src.model_utils import *


def clear_logs(model_architecture):
    """
    Clears the previous logs for the given model_architecture

    Args:
        model_architecture (string): model architecture
    """

    f = open(get_train_log_path(model_architecture), "w+")
    f.close()
    f = open(get_summary_path(model_architecture), "w+")
    f.close()


def get_train_log_path(model_architecture):
    """
    Path where the training logs should be saved. It may overwrite previous train logs of the same model_architecture.

    Args:
        model_architecture (string): model architecture

    Returns:
        string: path for the train log
    """

    return f'logs\\{model_architecture}_train_log.csv'


def get_summary_path(model_architecture):
    """
    Path where the summary log should be saved. It may overwrite previous summary logs of the same model_architecture.

    Args:
        model_architecture (string): model architecture

    Returns:
        string: path for the summary log
    """

    return f'logs\\{model_architecture}_summary.txt'


def get_trained_model_path(model_architecture):
    """
    Path to where a trained model should be saved. Its name will be unique to the current time.

    Args:
        model_architecture (string): model architecture

    Returns:
        string: path for the trained model
    """

    return f'trained_models\\{model_architecture}\\{model_architecture}_Model_{dt.now().strftime("%m_%d_%Y_%H_%M_%S")}'


def summary_log(model, model_architecture, x_test, y_test, trial, history, hyperparameters):
    """
    Writes a log which describes the properties of a trained model. 
    It shows, for example, the model structure and how it performs on test data. 
    It also saves the predicted targets based on the test data into a file called 'prediction.csv'

    Args:
        model (keras.Model): keras Model.
        model_architecture (string): Model architecture.
        x_test (ndarray): Test feature data.
        y_test (ndarray): Test target data.
        trial (int): Trial of the hyperparameter optimization.
        history (ndarray): Training history.
        hyperparameters (ndarray): Hyperparameters used in the model.
    """

    with open(get_summary_path(model_architecture), 'a') as f:
        f.write(f'{dt.now().strftime("%m_%d_%Y_%H_%M_%S")}. Model_architecture: {model_architecture}. Trial: {trial}. Epochs: {len(history.history["loss"])}. Hyperparameters: {hyperparameters}\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))

        if model_architecture == 'LSTM' or model_architecture == 'GRU':
            f.write(f'Evaluation: {model.evaluate(x_test)} \n')
        else:
            f.write(f'Evaluation: {model.evaluate(x_test, y_test)} \n')

        prediction = model.predict(x_test)
        print(prediction)
        print(y_test.to_numpy(dtype=float))
        f.write(f'Prediction: {prediction} \n')
        CAPE = cumulative_absolute_percentage_error(y_test.to_numpy(dtype=float), prediction.flatten())
        f.write(f'Cumulative Absolute Percentage Error on test data: {CAPE} \n\n\n\n')
        
        print(f'\nTrial finished. Test CAPE: {CAPE}\n\n\n')
