from keras.models import Model
from keras.optimizers import Adam, SGD
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from model import AttentionModel
from data_preprocessor import preprocess_data, tokenize, ids_to_keys, oh_2d
from config import params_config as cfg

import random
import json
import os

config =  tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)


def get_data():
    with open('../data/Time Dataset.json') as f:
        dataset = json.load(f)
    with open('../data/Time Vocabs.json') as f:
        human_vocab, machine_vocab = json.load(f)

    return dataset, human_vocab, machine_vocab

dataset, human_vocab, machine_vocab = get_data()

m = len(dataset)
human_vocab_size = len(human_vocab)
machine_vocab_size = len(machine_vocab)

X, Y, Xoh, Yoh =  preprocess_data(dataset,human_vocab,machine_vocab,cfg['Tx'],cfg['Ty'])

train_size =int(0.8 * m)
Xoh_train = Xoh[:train_size]
Yoh_train = Yoh[:train_size]
Xoh_test = Xoh[train_size:]
Yoh_test = Yoh[train_size:]

def train():
    model = AttentionModel(cfg,human_vocab_size,machine_vocab_size).model
    op = Adam(lr=cfg['learning_rate'],decay=cfg['decay'],clipnorm=cfg['clipnorm'])

    if os.path.exists('./Model/model.h5'):
        print('loading model ...')

        model.load_weights('./Model/model.h5')

        model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])

    else:
        model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])

        outputs_train = list(Yoh_train.swapaxes(0,1))
        model.fit(Xoh_train,outputs_train,epochs=cfg['epochs'],batch_size=cfg['batch_size'])

        if not os.path.exists('./Model'):
            os.mkdir('./Model')
        model.save_weights('./Model/model.h5')
    return model

model = train()
outputs_test = list(Yoh_test.swapaxes(0,1))
score = model.evaluate([Xoh_test],outputs_test)
print('Test loss:',score[0])


i = random.randint(0,m)

def get_prediction(model,x):
    prediction = model.predict(x)
    max_prediction = [y.argmax() for y in prediction]
    str_predict = ''.join(ids_to_keys(max_prediction,machine_vocab))
    return max_prediction,str_predict

print(type(Xoh[i]))                                               # 这里是np.ndarray
print(Xoh[i:i + 1].shape)
print(Xoh.shape)
max_prediction, str_prediction = get_prediction(model,Xoh[i:i+1]) # todo 注意：在这里的必须使用[i:i+1] 而不是[i],不然维度对应不上

print('Input:',str(dataset[i][0]))
print('Prediction text:',str(str_prediction))


'''
Xoh[i:i+1]的输出：
[[[0. 0. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  [0. 0. 1. ... 0. 0. 0.]
  ...
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 1.]]]
  
Xoh[i]的输出：
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
'''