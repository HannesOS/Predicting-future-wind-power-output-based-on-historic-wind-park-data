# Using TensorFlow to predict future wind power output based on historic wind park data - Bremen Big Data Challenge 2018 -
## Hannes Erbis and Marty Schüller
## Osnabrück Osnabrück
## Implementing ANNs with Tensorflow
### Project report

-- ABSTRACT -- 


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




