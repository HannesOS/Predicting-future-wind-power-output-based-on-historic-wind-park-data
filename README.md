# <p style="text-align:center"> Using TensorFlow to predict future wind power output based on historic wind park data - Bremen Big Data Challenge 2018 - </p>
## <p style="text-align:center"> <a href= "mailto:herbis@uni-osnabrueck.de">Hannes Erbis</a> and <a href= "mailto:mschueller@uni-osnabrueck.de">Marty Schüller</a> </p>
## <p style="text-align:center"> Osnabrück Osnabrück </p>
## <p style="text-align:center"> Implementing ANNs with Tensorflow </p>
### <p style="text-align:center"> Project report </p>
<br> <br>

To maintain a functional energy supply chain it is crucial to forecast future power outputs of generators. In this project we retroactively partake in the Bremen Big Data Challenge 2018 in which competitors are ought to predict the energy output of a wind park given weather measurements like wind speed. Our approach utilizes three different artificial neural network architectures: regular feedforward networks, LSTMs and GRUs. All models are realized with TensorFlow. Our results suggest that regular 'vanilla' feedforward networks consisting of four to five connected dense layers with 32 to 1042 neurons in each layer in combination with the use of an Autoencoder to reduce the dimensions of the data yield the most promising results. It is, however, important to note that a properly optimized LSTM or GRU could lead to an even better performance. Furthermore, we find out that using an Autoencoder to reduce the dimensionality of the feature space can have a positive impact on training performance regarding the speed at which the loss is minimized during backpropagation.
Our best model is able to predict the wind parks power output for the second half of 2017 with a cumulative absolute percentage error of 0.07302, after training on weather data from 2016 to mid 2017.




<br><br><br>
## Repository Overview: <br><br>
<b>architecture:</b> Contains subclasses inhereting from tf.keras.Model. Those are the models we are training. <br><br>
<b>data:</b> Train and test data taken from the Bremen Big Data Challenge 2018 (https://bbdc.csl.uni-bremen.de/index.php/2018h/23-aufgabenstellung-2018). <br><br>
<b>data_statistics:</b> Some statistics about the data. For example correlations between the variables. Seperated into test and train data statistics.  <br><br>
<b>figures:</b> Figures used in our report. <br><br>
<b>logs:</b> Training logs and model summaries when searching for hyperparameters. <br><br>
<b>predictions:</b> Wind energy output predictions the best model from each architecture. <br><br>
<b>src:</b> Some util modules that provide useful function: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    - <b>data-utils.py: Provides functions used for data processing and visualization.</b> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    - <b>model-utils.py: Provides functions relating to the model architectures and more.</b> <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    - <b>serialization-utils.py: Provides logging and model serialization functions.</b> <br><br>
<b>trained_models:</b> Serialized version of trained models.<br><br>
<b>energy_output_prediction.py:</b> Main module for this project. Demonstrates our best models and shows their results.<br><br>
<b>train.py:</b> Contains functions required for training and optimizing our models. If used as a main module, trains the specified model.<br><br>



