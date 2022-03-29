# Using TensorFlow to predict future wind power output based on historic wind park data - Bremen Big Data Challenge 2018 -
## Hannes Erbis and Marty Schüller
## Osnabrück Osnabrück
## Implementing ANNs with Tensorflow
### Project report

-- ABSTRACT -- 


## Repository Overview: <br><br>
architecture: Contains subclasses inhereting from tf.keras.Model. Those are the models we are training. <br>
data: Train and test data taken from the Bremen Big Data Challenge 2018 (https://bbdc.csl.uni-bremen.de/index.php/2018h/23-aufgabenstellung-2018). <br>
data_statistics: Some statistics about the data. For example correlations between the variables. Seperated into test and train data statistics.  <br>
figures: Figures used in our report. <br>
logs: Training logs and model summaries when searching for hyperparameters. <br>
predictions: Wind energy output predictions the best model from each architecture. <br>
src: Some util modules that provide useful function: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;   - data-utils: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    - model-utils: <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;    - serialization-utils <br>



