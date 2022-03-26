import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class CustomAutoencoder(Model):
    """
    Autoencoder with a custom dimension for the embedding layer. Once initiated, the encoder and decoder can be accessed independently from each other.
    """

    def __init__(self, latent_dim, n_features):
        """
        Initialize the autoencoder.

        Args:
            latent_dim (int): Dimension of the embedding layer.
            n_features (int): Number of features of the input data.
        """
        super(CustomAutoencoder, self).__init__()
        self.latent_dim = latent_dim   

        self.encoder = tf.keras.Sequential([
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(512, activation='relu'),
            Dense(n_features, activation='linear'),
        ])

    @tf.function
    def call(self, x):
        """
        Propagates data through the network (encoder and decoder).

        Args:
            x (ndarray): input data.

        Returns:
            ndarray: network output.
        """
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
