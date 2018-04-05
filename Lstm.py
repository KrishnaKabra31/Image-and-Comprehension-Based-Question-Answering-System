from keras.models import Sequential
from keras.layers import Dense, Activation,Merge,RepeatVector,concatenate
from keras.layers import LSTM
from keras import backend as tf
def Lstm(word_feature_size = 300,number_of_hidden_units = 1024,max_length = 30):
    left = Sequential()
    left.add(LSTM(1024, input_shape = (300,30),return_sequences=True))
    left.add(Dense(1024,activation = "softmax"))
    right = Sequential()
    right.add(left)
    right.add(LSTM(1024,return_sequences=False))
    print(tf.concatenate([tf.reshape(left.outputs[0][-1][-1],(1024,)),tf.reshape(right.outputs[0],(1024,))],axis=-1))
    right.add(Dense(1024,activation = "softmax"))
    return left,right
m,l = Lstm()
#print(l)
#print(m.outputs)
