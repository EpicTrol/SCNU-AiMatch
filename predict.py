
import numpy as np
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd 



from PIL import Image

import string
characters =  string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)


width, height, n_len, n_class = 120, 40, 4, len(characters)+1
#输入图片的大小，要识别的字符数，要识别的总类数



#---------------------以下为CNN+RNN++CTC LOSS实现-------------------------------------
from keras import backend as K


from keras.models import *
from keras.layers import *
from keras.optimizers import *
rnn_size = 128
l2_rate=1e-5
from keras.regularizers import *
input_tensor = Input((height, width, 3))
x = input_tensor
x = Lambda(lambda x:(x-127.5)/127.5)(x)

for i, n_cnn in enumerate([3, 4, 6]):
    for j in range(n_cnn):
        x = Conv2D(32*1**i, (3, 3), padding='same', kernel_initializer='he_uniform', 
                   kernel_regularizer=l2(l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
        x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

# x = AveragePooling2D((1, 2))(x)
cnn_model = Model(input_tensor, x, name='cnn')

input_tensor = Input((width, height, 3))
x = cnn_model(input_tensor)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[3]*conv_shape[2]

print (conv_shape, rnn_length, rnn_dimen)

x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2
rnn_imp = 0

x = Dense(2*rnn_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
x = BatchNormalization(gamma_regularizer=l2(l2_rate), beta_regularizer=l2(l2_rate))(x)
x = Activation('relu')(x)
#x = Dropout(0.2)(x)

gru_1 = GRU(2*rnn_size, implementation=rnn_imp, return_sequences=True, name='gru1')(x)
gru_1b = GRU(2*rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(2*rnn_size, implementation=rnn_imp, return_sequences=True, name='gru2')(gru1_merged)
gru_2b = GRU(2*rnn_size, implementation=rnn_imp, return_sequences=True, go_backwards=True, name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])

#x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax', kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])
base_model = Model(inputs=input_tensor, outputs=x)
model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])

#-------------------------------------------------以下为读入数据的代码 gen2为测试数据读入--------------
class Dataloader():
    def __init__(self,test_size):
        self.iter=0

        self.test_size = test_size

        file=[ './train/'+str(i)+'.jpg' for i in range(1,self.test_size+1) ]
        file=np.array(file)
        self.lb_file=file.reshape((self.test_size,1))


    def gen2(self,batch_size=10):
        X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)

        while True:
            end=(batch_size+1+self.iter*batch_size)
            begin=(1+self.iter*batch_size)
            for i in range(begin,end):

                i1=(i-1)%batch_size

                print(self.lb_file[i-1][0])
                temp_img = Image.open(str(self.lb_file[i-1][0]))
                temp_img = temp_img.resize((height,width), Image.BICUBIC)

                X[i1] = np.array(temp_img)

            self.iter=(self.iter+1)%int(int(self.test_size)/batch_size)
            yield [X, y, np.ones(batch_size)*rnn_length, np.ones(batch_size)*n_len], np.ones(batch_size)

def evaluate(batch_size=10, steps=500):
    data_loader = Dataloader(test_size)
    batch_acc = 0
    generator = data_loader.gen2(batch_size)
    result = []
    for i in range(steps):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :n_len]
        for p in range (0,steps):
            result.append(''.join([characters[x] for x in out[p]]))
        #print(out)
    return result
from keras.callbacks import *





model=load_model('./logs/base_modelrnn_shengcheng4_tune-acc-6.hdf5',custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
base_model=Model(inputs=model.get_layer('input_2').input ,outputs=model.get_layer('dense_2').output)

model.summary()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-3))

test_size = 5000  #需要测试多少张图片 这里就填多少张
batch_size = 10   #每个batch测试多少张图片，设备不好可以改小一点,但 test_size/batch_size_size一定要为整数

def Evaluator(test_size,batch_size):
    result = evaluate(batch_size,int(test_size/batch_size))
    result = np.array(result)
    result = result.reshape((test_size,1))
    file=[ str(i)+'.jpg' for i in range(1,test_size+1) ]
    file=np.array(file)
    file=file.reshape((test_size,1))
    array = np.concatenate((file , result) , axis=1)

    array = pd.DataFrame(array , columns=['ID' , 'label'])
    array.to_csv('submmision3.csv' , index=False)

    return result


Evaluator(test_size,batch_size)


