import pickle
import re
import sys
import math

import tensorflow as tf
import numpy as np
from Batch import BatchGenerator
from model import Model
from utils import *



with open('../data/renmindata.pkl') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)

print('train len: {}'.format(len(x_train)))
print('test len: {}'.format(len(x_test)))
print('word2id len: {}'.format(len(word2id)))
print('Create the data generator....')
data_train = BatchGenerator(x_train,y_train,shuffle=True)
data_valid = BatchGenerator(x_valid,y_train,shuffle=True)
data_test = BatchGenerator(x_test,y_train,shuffle=True)
print('Finished craating the data generator')

epoches = 31
batch_size = 32
config = {}
config['learning_rate'] = 1e-3
config['embedding_dim'] = len(x_train)
config['sen_len'] = len(x_train)
config['batch_size'] = batch_size
config['embedding_size'] = len(word2id) + 1 # todo +1 ？？？
config['tag_size'] = len(tag2id)
config['pretrained'] = False



def pretrained():
    embedding_pre = []
    print('use ptrtrained embedding')
    config['pretrained'] = True
    word2vec = {}
    with open('vec.txt','r',encoding='utf8') as input_data:
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])
    unknow_pre = []
    unknow_pre.extend([1] * 100)
    embedding_pre.append(unknow_pre)
    for word in word2id:
        if word2vec.get(word,None):
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)
    embedding_pre = np.asarray(embedding_pre)

def test():
    embedding_pre = []
    print('begin to test...')
    model = Model(config,embedding_pre,dropout_keep=1.)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print('Model not found ,please train your model first')
        else:
            path = ckpt.model_checkpoint_path
            print('loading pre-trained model from {}'.format(path))
            saver.restore(sess,path)
            test_input(model,sess,word2id,id2tag,batch_size)

def to_extraction():
    embedding_pre = []
    print('begin to extrction ...')
    model = Model(config,embedding_pre,dropout_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver =  tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print('Model not found ,please train yuout model first ')
        else:
            path = ckpt.model_checkpoint_path
            print('loading model from {}'.format(path))
            saver.restore(sess,path)
            extraction(sys.argv[1],sys.argv[2],model,sess,word2id,id2tag,batch_size)

def train():
    embedding_pre = []

    model = Model(config,embedding_pre,dropoiut_keep=0.5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train(model,sess,saver,epoches,batch_size,data_train,data_test,id2word,id2tag)