from keras.layers import LSTM,Bidirectional,Permute,Input,Dot,Lambda,Concatenate,RepeatVector,Dense,Activation,Softmax
import keras.backend as K
from keras.models import Model

class AttentionModel:
    def __init__(self,config,human_vocab_size,machine_vocab_size):

        # basic params
        self.Tx = config['Tx']
        self.Ty = config['Ty']

        self.x_vocab_size = human_vocab_size
        self.y_vocab_size = machine_vocab_size

        # net params
        self.layer1_size = config['layer1_size']
        self.layer2_size = config['layer2_size']

        # net func
        self.at_repeat = RepeatVector(self.Tx)  # 这层作用：
        self.at_concate = Concatenate(axis=-1)  # 这层作用：
        self.at_dense1 = Dense(8,activation='tanh')
        self.at_dense2 = Dense(1,activation='relu')
        self.at_softmax = Activation(lambda x:Softmax(axis=1)(x), name='attention_weights')
        self.at_dot = Dot(axes=1)

        self.layer3 = Dense(machine_vocab_size,activation=lambda x: Softmax(axis=1)(x))

        # get model
        self.model = self.get_model()

    def one_step_of_attention(self,h_prev,a):
        h_repeat = self.at_repeat(h_prev)
        i = self.at_concate([a,h_repeat])
        i = self.at_dense1(i)
        i = self.at_dense2(i)
        attention = self.at_softmax(i)
        context = self.at_dot([attention,a])

        return context

    def attention_layer(self,X,n_h,Ty):
        h = Lambda(lambda x:K.zeros(shape=(K.shape(x)[0],n_h)))(X)  # 这里的初始化的维度是二维的，因为return_sequence=False,不是所有的cell的h，这里的维度所以是[batch_size,hidden_size]
        c = Lambda(lambda x:K.zeros(shape=(K.shape(x)[0],n_h)))(X)

        at_LSTM = LSTM(n_h,return_state=True)
        output = []

        for _ in range(Ty):
            context = self.one_step_of_attention(h,X)

            h, _, c = at_LSTM(context,initial_state=[h,c])

            output.append(h)

        return output

    def get_model(self):
        X = Input(shape=(self.Tx,self.x_vocab_size))
        a1 = Bidirectional(LSTM(self.layer1_size,return_sequences=True),merge_mode='concat')(X)

        a2 = self.attention_layer(a1,self.layer2_size,self.Ty)

        a3 = [self.layer3(time_step) for time_step in a2]

        model = Model(inputs=X,outputs=a3)
        print(model.summary())

        return model