import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

class TextRNN:
    def __init__(self,num_classes,learn_rate,batch_size,decay_steps,decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):
        self.num_classes = num_classes
        self.batch = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = embed_size
        self.is_training = is_training
        self.learning_rate = learn_rate
        self.initializer = initializer

        # placement
        self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name='input_x')
        self.input_y = tf.placeholder(tf.int32,[None],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        self.global_step = tf.Variable(0,trainable=False,name='global_step')
        self.epoch_step = tf.Variable(0,trainable=False,name='epoch_step')
        self.epoch_increament = tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps,self.decay_rate = decay_steps, decay_rate

        # 定义所有权重
        self.instantiate_weights()
        # 定义主要的计算图
        self.logits = self.inference()
        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits,1,name='predictions')
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32),self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='Accuracy')

    def instantiate_weights(self):
        # 定义参数
        with tf.name_scope('embedding'):
            self.Embedding = tf.get_variable('Embedding',shape=[self.vocab_size,self.embed_size],initializer=self.initializer)
            self.W_projection = tf.get_variable('W_projection',shape=[self.hidden_size * 2,self.num_classes],initializer=self.initializer)
            self.b_projection = tf.get_variable('b_projection',shape=[self.num_classes])

    def inference(self):
        # 1.embeddign of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)

        # 2.Bi-LSTM layer
        # define lstm cell: get lstm cell output
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)

        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32)
        print('outputs:==>',outputs)
        output_rnn = tf.concat(outputs,axis=2)

        # 3.second LSTM layer
        rnn_cell = rnn.BasicLSTMCell(self.hidden_size * 2)
        if self.dropout_keep_prob is not None:
            rnn_cell = rnn.DropoutWrapper(rnn_cell,output_keep_prob=self.dropout_keep_prob)
        _, final_state_c_h = tf.nn.dynamic_rnn(rnn_cell,output_rnn,dtype=tf.float32)
        final_state = final_state_c_h[1]

        # 4.FC layer
        outputs = tf.layers.dense(final_state,self.hidden_size * 2,activation=tf.nn.tanh)

        # 5. logits(use linear layer)
        with tf.name_scope('output'):
            logits = tf.matmul(outputs,self.W_projection) + self.b_projection
        return logits

    def loss(self,l2_lambda=7e-3):
        with tf.name_scope('loss'):
            print(self.input_y)
            print(self.logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            loss = tf.reduce_mean(losses)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_loss
        return loss

    def train(self):
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val,global_step=self.global_step)
        return train_op

def test():
    num_classes = 10
    learning_rate = 1e-3
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1
    textRNN = TextRNN(num_classes,learning_rate,batch_size,decay_steps,decay_rate,sequence_length,vocab_size,embed_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros([batch_size,sequence_length])
            input_y = np.array([3,0,5,1,7,2,3,0])
            loss, acc, predict, _ = sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],
                                             feed_dict={textRNN.input_x:input_x,
                                                        textRNN.input_y:input_y,
                                                        textRNN.dropout_keep_prob:dropout_keep_prob})
            print('loss:{} acc:{} label:{} predict:{}'.format(loss,acc,input_y,predict))





if __name__ == '__main__':
    test()