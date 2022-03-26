import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

class CustomGRU(Model):
    """
    Stacked GATED recurrent unit ANN with a custom amount of single GRU entities, neurons in each GRU entity and an optional dropout layer.
    The output layer has a linear acitvation, which makes it useful for regression.
    """

    def __init__(self, layer_units, dropout_rate):
        """
        Initialize the (stacked) GRU.

        Args:
            layer_units (ndarray): Amount of neurons in each GRU entity. len(layer_units) determines the amount of single GRU entities in this model.
            dropout_rate (float): Dropout rate of the dropout layer. If dropout_rate is equal to 0 no dropout layer will be included.
        """

        super(CustomGRU, self).__init__()
        self.layers_ = []

        if len(layer_units) > 1:
            self.layers_.append(tf.keras.layers.GRU(layer_units[0], activation='relu', return_sequences=True))

        for i in range(len(layer_units)-1):
                self.layers_.append(tf.keras.layers.GRU(layer_units[i], activation='relu'))

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
