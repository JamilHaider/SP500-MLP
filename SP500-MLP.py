
# based on this blog post. Converted in Keras
# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import zipfile

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def plot_result(y_pred, y_test):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(y_test)
    line2, = ax1.plot(y_pred)
    plt.show()


zf = zipfile.ZipFile('data_stocks.zip', 'r')
zf.extract("data_stocks.csv")
data = pd.read_csv("data_stocks.csv")

data = data.drop(['DATE'], 1)

n = data.shape[0]
p = data.shape[1]

data = data.values

train_start = 0
train_end = int(np.floor(n * 0.8))
test_start = train_end + 1
test_end = n

data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
X_test = data_test[:, 1:]
Y_test = data_test[:, 0]

n_stocks = X_train.shape[1]


model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(n_stocks,)))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])

batch_size = 256
epochs = 20

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))


y_pred = model.predict(X_test)

plot_result(y_pred, Y_test)
