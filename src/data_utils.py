import numpy as np
import pandas as pd; pd.set_option('display.max_colwidth', 1); pd.set_option('display.max_columns', None)
import tensorflow as tf 
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from matplotlib import ticker
from datetime import datetime as dt

PATH_TRAIN_DATA = 'data\\train.csv'     #Relative path to the training data.
PATH_CHALLENGE_DATA = 'data\\challenge.csv'     #Relative path to the challenge data with the energy outputs being replaced with 'x'
PATH_SOLUTION = 'data\\eval.csv'        #Relative path to the challenge data with the actual energy outputs included.

#Orginal column names are in german. We want to translate them.
COLUMN_NAMES = ['Time', 'Wind speed at 48m [$m/s$]', 'Wind speed at 100m [$m/s$]', 'Wind speed at 152m [$m/s$]',
                'Wind direction at 48m', 'Wind direction at 100m', 'Wind direction at 152m',
                'Probabilistic wind speed at 100m (10 Percentile) [$m/s$]', 'Probabilistic wind speed at 100m (20 Percentile) [$m/s$]',
                'Probabilistic wind speed at 100m (30 Percentile) [$m/s$]', 'Probabilistic wind speed at 100m (40 Percentile) [$m/s$]',
                'Probabilistic wind speed at 100m (50 Percentile) [$m/s$]', 'Probabilistic wind speed at 100m (60 Percentile) [$m/s$]',
                'Probabilistic wind speed at 100m (70 Percentile) [$m/s$]', 'Probabilistic wind speed at 100m (80 Percentile) [$m/s$]',
                'Probabilistic wind speed at 100m (90 Percentile) [$m/s$]', 'Interpolated', 'Available capacity [$kW$]', 'Wind power output [$kWh$]']


def load_csv_data(path):
        """
        Load the data from a path into a Pandas DataFrame and extracts the date column.

        Args:
            path (string): Path to the data

        Returns:
            ndarray: the loaded data and the respective date column.
        """
        
        loaded_data = pd.read_csv(path)
        loaded_data.set_axis(COLUMN_NAMES, axis=1, inplace=True)
        date_time = pd.to_datetime(loaded_data.pop(COLUMN_NAMES[0])) #removing and saving the date column from the data

        return loaded_data, date_time


def get_features_and_targets(data=None, model_architecture='FFNN', excluded_features=[None], normalize=True, csv_path=None, autoencoder=None, window_size=5):
        """
        Extracts the features and targets of the data. 
        The target is the last column (energy output) while the other columns, with the data column being an exception, are the features.
        Some features can be exculed by the user.
        The user has the choice to normalize the data.
        This function can used with the data being pre-loaded, in which case it needs to be passed by the data argument. 
        If no data has been passed, a csv path has to be given from where the data can be accessed.

        Args:
            data (ndarray, optional): Entire data set. Defaults to None.
            model_architecture (str, optional): Which model architecture the user needs the data for. Defaults to 'FFNN'.
            excluded_features (list, optional): Indeces of features to be removed from the data set. Has to be given Defaults to [None].
            normalize (bool, optional): Whether or not the data should be normalized. Defaults to True.
            csv_path (string, optional): Path to the data. Defaults to None.
            autoencoder (keras.Model, optional): Autoencoder. Will be use to encode the features.
            window_size (int, optional): Size of data windows passed to the Keras TimeSeriesGenerator. Only used for LSTMs and GRUs Defaults to 5.

        Returns:
            ndarray: features and targets of the data set
        """
        if data is None:
            if csv_path is not None:
                    data = load_csv_data(csv_path)[0]

            else:
                    exit('Please specifiy the path to the data csv file (with parameter csv_path) or give the data directly (with paramter data)')
        feature_cols = data.columns[:-1]
        target_col = data.columns[-1]
        features = data.drop(feature_cols[[i for i in range(len(feature_cols)) if i in excluded_features]], axis=1).drop(target_col, axis=1) #Remove unwanted features
        targets = data[target_col]

        print(f'Succesfully loaded data with features {features.columns} and target {targets.name}.')
        if normalize:
                features = normalize_data(features)
        if autoencoder is not None:
            features = autoencoder.encoder.predict(features)
        
        if model_architecture == 'LSTM' or model_architecture == 'GRU':
                return TimeseriesGenerator(features, targets, window_size)
        return features, targets

        
def normalize_data(data):
        """
        Normalizes the data to bring each value into a -1 to 1 range using the formula x_norm = (x - x_mean)/x_std

        Args:
            data (ndarray): Entire data set

        Returns:
            ndarray: Normalized data. Has the same shape as the input data
        """

        return (data - data.mean()) / data.std()


def split_train_data(data, train_percent=0.8):
        """
        Splits the data set into train and validation data given a certain split percentage.

        Args:
            data (ndarray): Entire data set to split.
            train_percent (float, optional): Percentage of data which will be used as training data. Defaults to 0.8.

        Returns:
            ndarray: train data set and validation data set.
        """

        train_size = int(len(data) * train_percent)

        train_ds = data.take(train_size)
        valid_ds = data.skip(train_size)

        return train_ds, valid_ds


def analyze_data(data, save=True):
        """
        Prints out some statistics and useful information about the data.

        Args:
            data (ndarray): Entire data set
        """

        descr = data.describe()
        cov = data.cov().round(3)
        corr = data.corr().round(3)
        print(descr)
        print("\n\nCovariance matrix: \n", cov)
        print("\n\nCorrelation matrix: \n", corr)

        if save:
            descr.to_csv('data_statistics\description.csv')
            cov.to_csv('data_statistics\covariance.csv')
            corr.to_csv('data_statistics\correlation.csv')


def visualize_train_history(history):
        """
        Visualizes the training and validation loss aswell as the training and validation CAPE during the training of a model.


        Args:
            history (History): Training history of a model. Returned from using model.fit()
        """
        max_epochs = 100
        loss = history.history["loss"][:max_epochs]
        val_loss = history.history["val_loss"][:max_epochs]
        cape = history.history["cumulative_absolute_percentage_error_kerras_metric"][:max_epochs]
        val_cape = history.history["val_cumulative_absolute_percentage_error_kerras_metric"][:max_epochs]
        
        epochs = range(len(loss))[:max_epochs]
        
        fig, ax = plt.subplots(dpi=500, figsize=(7,4.5))
        
        ax.plot(epochs, loss, linewidth=2, color= "blue", label="Training loss")
        ax.plot(epochs, val_loss, linewidth=2, color="red", label="Validation loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (MSE)")
        ax.ticklabel_format(axis='y', style='sci', scilimits=(8,8), useMathText=True)
        ax.set_xlim(0, len(loss))
        ax.legend(frameon=False, shadow=True)
        fig.savefig("figures\loss.png")

        fig, ax = plt.subplots(dpi=500, figsize=(7,4.5))
        ax.plot(epochs, cape, linewidth=2, color="blue", label="Training CAPE")
        ax.plot(epochs, val_cape, linewidth=2, color="red", label="Validation CAPE")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("CAPE")
        ax.set_xlim(0, len(loss))
        ax.legend(frameon=False, shadow=True)
        fig.savefig("figures\CAPE.png")


def scatter_plot(data, col1, col2):
        """
        Creates a scatter plot between two variables and shows it.

        Args:
            data (ndarray): Entire data set
            col1 (string): Variable to show in scatter plot against col2
            col2 (string): Variable to show in scatter plot agains col1
        """

        fig, ax = plt.subplots(dpi=100, figsize=(5,5))
        ax.scatter(data[col1], data[col2], s=0.5, color='black')
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        plt.show()


def plot_time_series(data, date_time, column=COLUMN_NAMES[-1], start_day=330, end_day=344):
        """
        Creates a time series plots and saves it locally.

        Args:
            data (ndarray): Entire data set
            date_time (ndarray): Measurement times (x-values)
            column (string, optional): Column from where to extract the data from the data frame. Defaults to Wind power output.
            start_day (int, optional): First day of the time series the user wants to plot. Defaults to 330.
            end_day (int, optional): Last day of the time series the user wants to plot. Defaults to 344.
        """

        start_timestep = day_to_timestep(start_day) 
        end_timestep = day_to_timestep(end_day)
        
        fig, ax = plt.subplots(dpi=900, figsize=(9,3))
        ax.plot(date_time[start_timestep:end_timestep], data[column][start_timestep:end_timestep], color='black')
        ax.set_xlim((date_time[start_timestep], date_time[end_timestep]))
        ax.set_ylabel(column, fontsize=12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4), useMathText=True)
        fig.savefig(f'figures\\timeseries_{dt.now().strftime("%m_%d_%Y_%H_%M_%S")}.png')
        

def day_to_timestep(day):
        """
        Calculate the timestep of the beginning of a given day.
        Since there are 24 hours in a day and 4 measurements in an hour, the formula is: day * 24 * 4.

        Args:
            day (int): Day of which the user wants to know the first timestep of.

        Returns:
           int: time step of the beginning of a given day.
        """

        return day * 24 * 4 

if __name__ == "__main__":
    train_data, _ = load_csv_data(PATH_TRAIN_DATA)
    analyze_data(train_data)