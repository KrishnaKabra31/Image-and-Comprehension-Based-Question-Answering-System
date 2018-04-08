from keras.models import Sequential,Model
from keras.layers import Dense,Dot,Lambda,Input, Activation,Merge,RepeatVector,Concatenate,Dropout,Reshape
from keras.layers import LSTM
from keras import backend as tf
def Lstm(word_feature_size = 300,number_of_hidden_units = 1024,que_length = 26):#,num_words,embedding_dim):
  #  left = Sequential()
    inputs = Input(shape = (26,300))
    left = LSTM(1024, input_shape = (que_length,word_feature_size),return_sequences=True)(inputs)
    p = Lambda(lambda x:x[:,-1,:])(left)
    left = (Dropout(rate=0.4))(left)
    print(p)
    left = LSTM(1024,return_sequences=False)(left)
    left = Lambda(lambda x:x[:,:])(left)
    left = Dropout(rate = 0.4)(left)
    print(left)
    con = Concatenate(axis = 1)([p,left])
    model = Model(input= inputs, output=con)
    print(model.output_shape)
    return con
m = Lstm()
#print(l)
#print(m.outputs)
