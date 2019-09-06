from keras.optimizers import Adam
from keras.models import load_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

from data_preprocessor import preprocess_data,tokenize,ids_to_keys,oh_2d
from model import AttentionModel
from config import params_config


import random
import json
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)


def get_data():
    with open('./data/Time Dataset.json') as f:
        dataset = json.load(f)
    with open('./data/Time Vocabs.json') as f:
        human_vocab, machine_vocab =  json.load(f)

    return dataset, human_vocab, machine_vocab

dataset, human_vocab, machine_vocab = get_data()

# 获取长度
human_vocab_size = len(human_vocab)
machine_vocab_size = len(machine_vocab)
m = len(dataset)
X, Y, Xoh, Yoh = preprocess_data(dataset,human_vocab,machine_vocab,params_config['Tx'],params_config['Ty'])

# train_test_split
train_size  =  int(.8 * m)
Xoh_train = Xoh[:train_size]
Yoh_train = Yoh[:train_size]
Xoh_test  =  Xoh[train_size:]
Yoh_test  =  Yoh[train_size:]


def train():
    model = AttentionModel(params_config,human_vocab_size,machine_vocab_size).model

    op = Adam(lr=params_config['learning_rate'],decay=params_config['decay'],clipnorm=params_config['clipnorm'])

    if os.path.exists('./Model/model.h5'):
        print('loading model...')

        model.load_weights('./Model/model.h5')

        model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])


    else:
        model.compile(optimizer=op,loss='categorical_crossentropy',metrics=['accuracy'])

        outputs_train = list(Yoh_train.swapaxes(0,1))

        model.fit(Xoh_train,outputs_train,epochs=params_config['epochs'],batch_size=params_config['batch_size'],validation_split=0.1)

        if not os.path.exists('Model'):
            os.mkdir('Model')

        model.save_weights('./Model/model.h5')
    return model

def get_predict(model,x):
     prediction = model.predict(x)
     max_prediction = [y.argmax() for y in prediction]
     str_prediction = ''.join(ids_to_keys(max_prediction,machine_vocab))
     return max_prediction, str_prediction

if __name__ == '__main__':
    model = train()
    outputs_test = list(Yoh_test.swapaxes(0,1))
    score = model.evaluate([Xoh_test],outputs_test)
    print('Test loss:',score[0])

    i = random.randint(0,m-train_size-1)

    print(Xoh_test[i:i+1])

    max_prediction, str_prediction =  get_predict(model,Xoh[i:i+1])

    print('Input:',str(dataset[i][0]))

    print('Prediction text:',str_prediction)