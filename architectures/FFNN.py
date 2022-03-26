import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

class CustomFFNN(Model):
    """
    Regular Feed-Forward-Neural-Network consisting of Dense layers.
    The output layer has a linear acitvation, which makes it useful for regression.
    """

    def __init__(self, layer_units, dropout_rate):
        """
        Initialize the Feed-Forward Network.

        Args:
            layer_units (ndarray): Amount of neurons in each Dense layer. len(layer_units) determines the amount of hidden Dense layers in this model.
            dropout_rate (float): Dropout rate of the dropout layer. If dropout_rate is equal to 0 no dropout layer will be included.
        """
        super(CustomFFNN, self).__init__()
        self.layers_ = []
        for layer_unit in layer_units:
            self.layers_.append(Dense(layer_unit, activation='relu'))
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