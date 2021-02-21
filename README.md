# Project Title

Wind energy prediction for the La-Haute-Borne onshore wind farm (is released as a Building AI course project)

## Summary

The current project investigates data-driving methods to predict wind energy generation for the onshore "La Haute Borne" wind farm. The hybrid model was suggested to get short-term power forecasts using both historical in situ measurements available from ENGIE and global reanalysis data of MERRA-2. It was shown that adding three extra meteorological parameters - pressure, humidity, and temperature (Case 3) - allowed to reach a higher accuracy compared with cases when weather parameters were completely ignored (Case 1) or used partially (Case 2); this was proved by applying multivariate, one-step long short-term memory (LSTM) networks. Additionally, it was shown that the CNN-LSTM approach allowed to reach a better accuracy while predicting wind power for 12h and 24h ahead compared to the LSTM model.

## Background

2019 year showed a strong growth of renewable energy - the latter was able to increase its share by a record amount, accounting for over 40% of the growth in primary energy, mainly because of wind and solar power; in particular, wind provided the largest contribution to renewables growth - 1.4 EJ [BP](https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html).
Most likely this trend of renewables rise in energy portfolio will continue in the future. Therefore, we need to predict wind energy output in a fast and reliable manner.

The currect project aims to solve the following issues:
* Prove that adding more meteorological data might help to improve the quality of forecasts. It was demonstrated by applying multivariate, one-step long short-term memory (LSTM) networks. 
* Multivariate, multi-step deep learning networks were built for predicting wind power within 12h and 24h ahead. Here, CNN-LSTM network showed a higher accuracy compared with a case of applying LSTM network. 

## How is it used?

The project was implemented using [Python] (https://www.python.org/) as the main source for writing codes with Pandas, Numpy, Tensorflow with Keras. In particular, first it was required to read .csv files from ENGIE, merge them, preprocess data and then merge this dataset with data available from MERRA-2 project. This enriched dataset was then used for all wind forecasts (70% of data is a training set, 10% - a validation set, and remaining 20% - a test set).

Images will make your README look nice!
Once you upload an image to your repository, you can link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

## Data sources
Historical in situ measurements for the La-Haute-Borne wind park were available from [ENGIE](https://opendata-renewables.engie.com/) and global reanalysis data were taken as a part of MERRA-2 project from [NASA](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)

## Results

The main drawback of applying multivariate, one-step LSTM method is that in this case the whole test dataset is forecasted at once, which is not how it really happens. Thus, to improve the quality of the forecasts we need to apply multivariate, multi-step LSTM and CNN-LSTM scheme. 

This is a part of the code for applying LSTM method:

```
# Define the model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout #, Flatten, RepeatVector #Dropout
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D

model = Sequential()
model.add(LSTM(100, return_sequences = True,
               input_shape=(n_steps_in, n_features)))
#model.add(RepeatVector(n_steps_out))
model.add(LSTM(100))
model.add(Dense(n_steps_out))
```

And this is - a part of the code for CNN-LSTM 24h forecast:

```
# Define the model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
model = Sequential()
model.add(Conv1D(nb_filter=64, filter_length=3,
                 activation='relu', input_shape=(n_steps_in, n_features)))
#model.add(Conv1D(nb_filter=64, filter_length=3, activation='relu'))
#model.add(Conv1D(nb_filter=32, filter_length=4, activation='relu'))
#model.add(Conv1D(nb_filter=64, filter_length=3, activation='relu'))
model.add(MaxPooling1D(pool_length=6))
model.add(LSTM(180, activation='relu'))
model.add(Dense(n_steps_out))
```
Here n_steps_out = 24 (24 hour forecast). Table below is summarized some other parameters:

| Parameter        | Value       |
| ---------------- | ----------- |
| Iterations       | 30          |
| Layers LSTM      | 2           |
| Layers CNN-LSTM  | 3 + 1       |

## Challenges

The current project was only focused on the short-term forecasts, therefore, it did _not_ consider applications of data-driven methods for long-term wind predictions. This could be the next step of the project.

## What next?

In addition to application of data-driven methods for long-term forecasts, it can be interesting to investigate some other available deep learning techniques, e.g. ConvLSTM networks.

## Licence

If you are going to use [ENGIE](https://opendata-renewables.engie.com/) dataset to practise, please notice that it has its own [LICENCE](https://www.etalab.gouv.fr/wp-content/uploads/2017/04/ETALAB-Licence-Ouverte-v2.0.pdf).

## Acknowledgments

ENGIE Renewables is acknowledged for sharing their "La Haute Borne" dataset with researchers. It indeed helps to feel how difficult is to work with raw data. Gratitudes should also be sent to the Global Modeling and Assimilation Office (GMAO) at NASA Goddard Space Flight Center for the MERRA-2 reanalysis data which was provided through the NASA GES DISC online archive.
