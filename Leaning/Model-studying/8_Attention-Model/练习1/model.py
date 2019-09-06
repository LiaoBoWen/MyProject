from keras.layers import LSTM,Bidirectional,Dot,Concatenate,Input,RepeatVector,Dense,Activation,Lambda
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
import tensorflow as tf

from config import params_config

class AttentionModel:
    def __init__(self,config,human_vocab_size,machine_vocab_size):
        # params
        self.Tx = config['Tx']
        self.Ty = config['Ty']
        self.layer1_size = config['layer1_size']
        self.layer2_size = config['layer2_size']

        self.x_vocab_size = human_vocab_size
        self.y_vocab_size = machine_vocab_size

        # net func
        self.at_repeat = RepeatVector(self.Tx)
        self.at_concatenate = Concatenate(axis=-1)
        self.at_dense1 = Dense(8,activation='tanh')     # 这里的tanh函数效果不错
        self.at_dense2 = Dense(1,activation='relu')     # 这里的relu效果也还行
        self.at_softmax = Activation(lambda x: K.softmax(x,axis=1),name='attention_weights') # 这里的softmax在网络层中有没有
        self.at_dot = Dot(axes=1)

        self.layer3 =Dense(machine_vocab_size,activation=lambda x: K.softmax(x,axis=1))

        self.model =  self.get_model()

    def one_step_of_attention(self,h_prev,a):
        h_repeat = self.at_repeat(h_prev)

        i = self.at_concatenate([a,h_repeat])
        i = self.at_dense1(i)
        i = self.at_dense2(i)
        attention = self.at_softmax(i)
        context = self.at_dot([attention,a])

        return context


    def attention_layer(self,X,n_h,Ty):
        # RNN 的初始化
        h = Lambda(lambda X:K.zeros(shape=(K.shape(X)[0],n_h)))(X)
        c = Lambda(lambda X:K.zeros(shape=(K.shape(X)[0],n_h)))(X)

        at_LSTM = LSTM(n_h,return_state=True)

        output = []

        for _ in range(Ty):

            context = self.one_step_of_attention(h,X)

            h, _, c = at_LSTM(context,initial_state=[h,c])

            output.append(h)

        return output


    def get_model(self):

        X = Input(shape=[self.Tx,self.x_vocab_size])

        a1 = Bidirectional(LSTM(self.layer1_size,return_sequences=True),merge_mode='concat')(X)

        a2 = self.attention_layer(a1,self.layer2_size,self.Ty)

        a3 = [self.layer3(time_step) for time_step in a2]

        model = Model(inputs=X,outputs=a3)

        return model