
from keras.callbacks import *
from keras.regularizers import *
from keras.optimizers import *
from keras.layers import *
from keras.models import *
from keras import backend as K
import numpy as np
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
'''
#进行配置，每个GPU使用60%上限现存
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

#os.environ["CUDA_VISIBLE_DEVICES"]="2"#使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

# 设置session
KTF.set_session(session)

from keras.utils import multi_gpu_model
'''

from PIL import Image

import string
characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
print(characters)


width, height, n_len, n_class = 120, 40, 4, len(characters)+1


# ---------------------以下为CNN+RNN++CTC LOSS实现-------------------------------------
# 参考https://github.com/ypwhs/baiduyun_deeplearning_competition
rnn_size = 128
l2_rate = 1e-4
input_tensor = Input((height, width, 3))
x = input_tensor
x = Lambda(lambda x: (x-127.5)/127.5)(x)

for i, n_cnn in enumerate([3, 4, 6]):
    for j in range(n_cnn):
        x = Conv2D(32*1**i, (3, 3), padding='same', kernel_initializer='he_uniform',
                   kernel_regularizer=l2(l2_rate))(x)
        x = BatchNormalization(gamma_regularizer=l2(
            l2_rate), beta_regularizer=l2(l2_rate))(x)
        x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

# x = AveragePooling2D((1, 2))(x)
cnn_model = Model(input_tensor, x, name='cnn')

input_tensor = Input((width, height, 3))
x = cnn_model(input_tensor)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[3]*conv_shape[2]

print(conv_shape, rnn_length, rnn_dimen)

x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2
rnn_imp = 0

x = Dense(2*rnn_size, kernel_initializer='he_uniform',
          kernel_regularizer=l2(l2_rate), bias_regularizer=l2(l2_rate))(x)
x = BatchNormalization(gamma_regularizer=l2(
    l2_rate), beta_regularizer=l2(l2_rate))(x)
x = Activation('relu')(x)
#x = Dropout(0.2)(x)

gru_1 = GRU(2*rnn_size, implementation=rnn_imp,
            return_sequences=True, name='gru1')(x)
gru_1b = GRU(2*rnn_size, implementation=rnn_imp,
             return_sequences=True, go_backwards=True, name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(2*rnn_size, implementation=rnn_imp,
            return_sequences=True, name='gru2')(gru1_merged)
gru_2b = GRU(2*rnn_size, implementation=rnn_imp, return_sequences=True,
             go_backwards=True, name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])

#x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax', kernel_regularizer=l2(
    l2_rate), bias_regularizer=l2(l2_rate))(x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [x, labels, input_length, label_length])
base_model = Model(inputs=input_tensor, outputs=x)
model = Model(inputs=[input_tensor, labels, input_length,
                      label_length], outputs=[loss_out])

'''
def gen(batch_size=128):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size)*rnn_length, np.ones(batch_size)*n_len], np.ones(batch_size)
'''


class Dataloader():
    def __init__(self):
        self.iter = 0
        self.iter2 = 0
        label = pd.read_csv('./train/train_label.csv')
        label = label['label']
        label = np.array(label)
        self.train_size = 20000  # 训练集共用2万张
        self.split = 0.85  # 0.95用来训练 0.05用来验证
        label = label.reshape((self.train_size, 1))
        file = ['./train/'+str(i)+'.jpg' for i in range(1, self.train_size+1)]
        file = np.array(file)
        file = file.reshape((self.train_size, 1))
        lb_file = np.hstack((file, label))
        np.random.shuffle(lb_file)
        self.lb_file = lb_file

    def gen2(self, batch_size=10):
        X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)

        while True:
            end = (batch_size+1+self.iter*batch_size)
            begin = (1+self.iter*batch_size)
            for i in range(begin, end):

                i1 = (i-1) % batch_size
                random_str = str(self.lb_file[i-1][1])

                temp_img = Image.open(str(self.lb_file[i-1][0]))
                temp_img = temp_img.resize((height, width), Image.BICUBIC)

                X[i1] = np.array(temp_img)
                y[i1] = [characters.find(x) for x in random_str]
            self.iter = (
                self.iter+1) % int(int(self.train_size*self.split)/batch_size)
            yield [X, y, np.ones(batch_size)*rnn_length, np.ones(batch_size)*n_len], np.ones(batch_size)

    def gen3(self, batch_size=10):
        X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
        y = np.zeros((batch_size, n_len), dtype=np.uint8)

        while True:
            end = (batch_size+int(self.train_size *
                                  self.split)+self.iter2*batch_size)
            begin = (int(self.train_size*self.split)+1+self.iter2*batch_size)
            for i in range(begin, end):

                i1 = (i-1) % batch_size
                random_str = str(self.lb_file[i-1][1])

                temp_img = Image.open(str(self.lb_file[i-1][0]))
                temp_img = temp_img.resize((height, width), Image.BICUBIC)

                X[i1] = np.array(temp_img)
                y[i1] = [characters.find(x) for x in random_str]
            self.iter2 = (self.iter2+1) % int((self.train_size -
                                               int(self.train_size*self.split))/batch_size)
            yield [X, y, np.ones(batch_size)*rnn_length, np.ones(batch_size)*n_len], np.ones(batch_size)


def evaluate(batch_size=10, steps=2):
    batch_acc = 0
    generator = data_loader.gen3(batch_size)
    for i in range(steps):
        [X_test, y_test, _, _], _ = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(
            y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :n_len]
        if out.shape[1] == n_len:
            batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / steps


#model=load_model('modelrnn_shengcheng4-1340-0.19.hdf5',custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
base_model = Model(inputs=model.get_layer('input_2').input,
                   outputs=model.get_layer('dense_2').output)
data_loader = Dataloader()
model.summary()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
              optimizer=Adam(1e-3))


class Evaluator(Callback):
    def __init__(self):
        self.acc_max = 0

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(steps=3)*100
        if acc > self.acc_max:
            self.acc_max = acc
            model.save("./logs/base_modelrnn_shengcheng4_tune-acc-" +
                       str(int(acc))+".hdf5")
            print(acc, "save base_modelrnn_shengcheng4_tune-acc:"+str(acc)+".hdf5")
        else:
            print('acc: %f%%' % acc)


evaluator = Evaluator()

callback_lists = [evaluator]
model.fit_generator(data_loader.gen2(10), steps_per_epoch=30, epochs=20000,
                    callbacks=callback_lists, validation_data=data_loader.gen3(10), validation_steps=1)
