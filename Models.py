from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras import Input
# keras also requires tensorflow "pip3 install tensorflow"

def createLSTMModel():
    model = Sequential()
    # not using relu as an activation function hear because of the unstableness it brings
    model.add(LSTM(units=64, input_shape=(33,1)))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mean_squared_error'])

    return model

def createConvolutionalModel():
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(33,1)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mean_squared_error'])

    return model

def createLinearModel():
    model = Sequential()
    model.add(Dense(100, input_dim=33, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mean_squared_error'])

    return model