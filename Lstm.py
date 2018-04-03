from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

def LSTM(word_feature_size = 300,number_of_hidden_units = 1024,max_length = 30):
    model = Sequential()
    model.add(LSTM(output_size = number_of_hidden_units,return_sequences = True,input_shape=(max_length,word_feature_size)))
    model.add(LSTM(output_size = number_of_hidden_units,return_sequences = False))
    return model


    
