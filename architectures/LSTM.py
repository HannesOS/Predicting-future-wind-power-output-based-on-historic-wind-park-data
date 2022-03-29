import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

class CustomLSTM(Model):
    """
    Stacked Long short-term memory ANN with a custom amount of single LSTM entities, neurons in each LSTM entity and an optional dropout layer.
    The output layer has a linear acitvation, which makes it useful for regression.
    """

    def __init__(self, layer_units, dropout_rate):
        """
        Initialize the (stacked) LSTM.

        Args:
            layer_units (ndarray): Amount of neurons in each LSTM entity. len(layer_units) determines the amount of single LSTM entities in this model.
            dropout_rate (float): Dropout rate of the dropout layer. If dropout_rate is equal to 0 no dropout layer will be included.
        """

        super(CustomLSTM, self).__init__()
        self.layers_ = []

        for i in range(len(layer_units)-1):
            self.layers_.append(tf.keras.layers.LSTM(layer_units[i], activation='relu', return_sequences=True))

        self.layers_.append(tf.keras.layers.LSTM(layer_units[-1], activation='relu'))

        if dropout_rate > 0:
            self.layers_.append(Dropout(dropout_rate))
        self.layers_.append(Dense(1, activation='linear'))


    @tf.function
    def call(self, x):
        """
        Propagates data through the network.

        Args:
            x (ndarray): input data.

        Returns:
            ndarray: network output.
        """
        
        for layer in self.layers_:
            x = layer(x)
        return x
