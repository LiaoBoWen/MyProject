from keras.layers import Bidirectional,Concatenate,Permute,Dot,Input,LSTM,Multiply,Embedding,Reshape,RepeatVector, Dense,Activation,Lambda,Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler
import keras.backend.tensorflow_backend as KTF
import keras.backend as K       # todo 后端
import tensorflow as tf

from preprocessor import preprocess_data,tokenize,ids_to_keys,oh_2d
# from plot_attention import plot_attention_graph

import matplotlib.pyplot as plt
import numpy as np

import random
import json
import os


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      # 设置动态分配内存
session = tf.Session(config=config)
KTF.set_session(session)



with open('./data/Time Dataset.json') as f:
    dataset = json.load(f)
with open('./data/Time Vocabs.json') as f:
    human_vocab, machine_vocab = json.load(f)

human_vocab_size = len(human_vocab)
machine_vocab_size = len(machine_vocab)

# number of training examples
m = len(dataset)


# tokenize the data using vocabularies
Tx = 41 # Max x sequence length
Ty = 5 # y sequence length
X, Y, Xoh, Yoh = preprocess_data(dataset,human_vocab,machine_vocab,Tx,Ty)

# Split dat 80-20 between training and test
train_size = int(0.8 * m)
Xoh_train = Xoh[:train_size]
Yoh_train = Yoh[:train_size]
Xoh_test = Xoh[train_size:]
Yoh_test = Yoh[train_size:]

# Check the code works:
# i = 4
# print('Input data point',str(i),'.\n')
# print('The data input is :',str(dataset[i][0]))
# print('The data output is :',str(dataset[i][1]))
# print()
# print('The tokenized input is :',str(X[i]))
# print('The tokenized output is :',str(Y[i]))
# print()
# print('The one-hot input is :',str(Xoh[i]))
# print('The one-hot output is :',str(Yoh[i]))


'''define attention mechanism'''
# Define some model metadata
layer1_size = 32                            # todo 每一个门（3个sigmoid和1个tanh）都算一个前馈神经网络，这里的num_unit是每个前馈网络层的隐藏元个数  https://blog.csdn.net/xiewenbo/article/details/79452843
layer2_size = 64    # Attention Layer

# Define part of the attention layer globally so as to share the same layers for each attention step
def softmax(x):
    return K.softmax(x,axis=1)

at_repeat = RepeatVector(Tx)                    # todo 因为h和a1维度不一致的问题
at_concatenate = Concatenate(axis=-1)
at_dense1 = Dense(8,activation='tanh')          # todo 在relu、sigmoid、tanh中tanh表现的最好
at_dense2 = Dense(1,activation='relu')          # todo 防止多层导致梯度消失
at_softmax = Activation(softmax,name='attention_weights')   #todo 和吴恩达讲的softmax不一样
at_dot = Dot(axes=1)    # todo 点积


def one_step_of_attention(h_prev,a):
    '''
    :param h_prev: Previous hidden state of a RNN layer (m,n_ h)
    :param a: Input data ,possibly processed (m,Tx,n_a) todo 这里的维度问题：m是什么的维度 ？n_a是什么的维度？
    :return:Current context(m,Tx,n_a)
    '''
    # Repeat vector to match a's dimensions
    h_repeat = at_repeat(h_prev)
    # Calculate attention weights
    i = at_concatenate([a,h_repeat])
    i = at_dense1(i)                     # todo 这三步就是使用的吴恩达的说的放入神经网络进行训练的过程
    i = at_dense2(i)
    attention = at_softmax(i)
    # Calculate the context
    context = at_dot([attention,a])     # todo 权重开始分配  点积。。。

    return context

def attention_layer(X,n_h,Ty):
    '''
    Create an attention layer
    :param X: Layer input (m,Tx,x_vocab_size)
    :param n_h: Size of LSTM hidden layer
    :param Ty: Timesteops in output sequence
    :return: Output of the attention layer (m,Tx,n_h)
    '''
    # Define the default state for the LSTM layer
    h = Lambda(lambda X:K.zeros(shape=(K.shape(X)[0],n_h)))(X)      # 初始化h，c
    c = Lambda(lambda X:K.zeros(shape=(K.shape(X)[0],n_h)))(X)

    at_LSTM = LSTM(n_h,return_state=True)


    output = []

    # Run attention step and RNN for each output time step          # todo Decoder
    for _ in range(Ty):
        context = one_step_of_attention(h,X)
        # 这里在注意力的技术使用了c之
        h, _, c = at_LSTM(context,initial_state=[h,c])      # todo 传入的是h，同时也只有第一遍c、h都是0

        output.append(h)

    return output

layer3 = Dense(machine_vocab_size,activation=softmax)

def get_model(Tx,Ty,layer1_size,layer2_size,x_vocab_size,y_vocab_size):
    '''
    :param Tx: Number of x timesteps
    :param Ty: Number of y timesteps
    :param layer1_szie: Number of neurons in BiLSTM
    :param layer2_size: Number of neurous in attention LSTM hidden layer
    :param x_vocab_size: Number of possible token types for x
    :param y_vocab_size: Number of possible token types for y
    :return:
    '''

    # Create layers one by one
    X = Input(shape=(Tx,x_vocab_size))    # todo keras自动到第一维度添加None象征数据大小

    a1 = Bidirectional(LSTM(layer1_size,return_sequences=True),merge_mode='concat')(X)      # todo Encoder   输出shape?

    a2 = attention_layer(a1,layer2_size,Ty)         # todo Decoder

    a3 = [layer3(time_step) for time_step in a2]

    # Create Keras model
    model = Model(inputs=X,outputs=a3)

    return model



def train():
    model = get_model(Tx,Ty,layer1_size,layer2_size,human_vocab_size,machine_vocab_size)

    op = Adam(lr=0.05,decay=0.04,clipnorm=1.)       # todo clipnorm用来对梯度进行约束，防止梯度爆炸，超过的部分按照公式计算：t * clip_norm / l2norm(t)

    if os.path.exists('./Model/model.h5'):
        print('loading model...')
        model.load_weights('./Model/model.h5')

        model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])
    else:

        model.compile(optimizer=op, loss='categorical_crossentropy', metrics=['accuracy'])  # todo 损失计算方法选择这个的原因：

        outputs_train = list(Yoh_train.swapaxes(0, 1))      # todo swapaxes交换维度，可以debug测试一下a3的shape就知道

        model.fit(Xoh_train,outputs_train,epochs=5,batch_size=100,validation_split=0.1)  # 开始训练！！

        if not os.path.exists('Model'):
            os.mkdir('Model')
        model.save_weights('./Model/model.h5')

    return model

# Evaluate the test performance
model = train()
outputs_test = list(Yoh_test.swapaxes(0,1))
score = model.evaluate([Xoh_test],outputs_test)
print('Test loss:',score[0])



################################################
################################################
# Run it through our model

i = random.randint(0,m)

def get_prediction(model,x):
    prediction = model.predict(x)
    max_prediction = [y.argmax() for y in prediction]
    str_prediction = ''.join(ids_to_keys(max_prediction,machine_vocab))
    return (max_prediction, str_prediction)

# print(Xoh[i:i+1])
max_prediction, str_prediction = get_prediction(model,Xoh[i:i+1])

print('Input :',str(dataset[i][0]))
# print('Tokenized :',str(X[i]))
# print('Prediction :',str(max_prediction))
print('Prediction text :',str(str_prediction))


# plot attention
# plot_attention_graph(model,dataset[i][0],Tx,Ty,human_vocab)
