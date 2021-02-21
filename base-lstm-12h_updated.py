import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, explained_variance_score
from math import sqrt

np.set_printoptions(threshold = np.inf)

# Data loading
df = pd.read_csv('final-data.csv', sep=',')
df['Date_time'] = pd.to_datetime(df['Date_time'])
df.sort_values(by=['Date_time'], inplace=True, ascending=True)
df.set_index('Date_time', inplace=True)
df = df[['PS','QV10M','Ot','Wa','Ws','P']] #PS - pressure, QV10M - humidity, Ot - temperature, Wa - wind direction, Ws - wind speed, P - power
#print(df)

# Train-test split
train_size = int(len(df)*.7) #70-10-20 split
val_size = int(len(df)*.1)
test_size = len(df) - train_size - val_size
train, val, test = df.values[0:train_size,:], df.values[train_size:(train_size+val_size),:], df.values[(train_size+val_size):len(df),:]
#print(train)
#print(val)
#print(test)

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of the pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # Check if we are beyond the dataset
        if out_end_ix > len(sequences):
                break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1],sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 24, 12 

# Convert into input/output
trainX, trainY = split_sequences(train, n_steps_in, n_steps_out)
valX, valY = split_sequences(val, n_steps_in, n_steps_out)
testX, testY = split_sequences(test, n_steps_in, n_steps_out)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)
n_features = trainX.shape[2]
print(testY)

# Define the model
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, return_sequences = True,
               input_shape=(n_steps_in, n_features)))

model.add(LSTM(100))
model.add(Dense(n_steps_out))
model.summary()
model.compile(optimizer='adam', loss='mse')
history = model.fit(trainX, trainY,
                    nb_epoch=30, batch_size=128,
                    validation_data=(valX, valY),
                    verbose=2, shuffle=False)
                    
# Predicting
predictions = model.predict(testX, verbose=2)
print(predictions)

# Data visualization & error metrics
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
fig,ax = plt.subplots(figsize=(14,4))
ax.plot(testY[0], label='True',marker='None',color='black')
ax.plot(predictions[0], label='Predicted',marker='o', color='navy')
ax.set_xlabel('Time')
ax.set_ylabel('Power [MW]')
plt.legend(loc='lower right')
plt.savefig('base-lstm-12h.jpg', dpi=1200, bbox_inches='tight')
#plt.show()

print("MSE: %.6f" % mean_squared_error(testY[0], predictions[0]))
print("MAE: %.6f" % mean_absolute_error(testY[0], predictions[0]))
