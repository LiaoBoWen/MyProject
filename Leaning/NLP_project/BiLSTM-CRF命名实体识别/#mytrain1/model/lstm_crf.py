import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

from model.basemodel import Base_model

class BiLstm_model(Base_model):
    def __init__(self,config):
        super().__init__(config)
        model_output = self.lstm_layer()
        self.logits = self.lstm_project(model_output)
        self.loss = self.loss_layer(self.logits,self.lengths)
        self.get_optimizer()

    def lstm_layer(self):
        fw_cell = rnn.LSTMCell(self.lstm_dim)
        bw_cell = rnn.LSTMCell(self.lstm_dim)

        # todo 这里的sequence_length
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,self.embedding,dtype=tf.float32,sequence_length=self.lengths)

        return tf.concat(outputs,axis=-1)

    def lstm_project(self,inputs):
        with tf.variable_scope('project'):
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W',dtype=tf.float32,shape=[self.lstm_dim * 2, self.lstm_dim],initializer=self.initializer)
                b = tf.get_variable('b',dtype=tf.float32,shape=[self.lstm_dim],initializer=self.initializer)

                lstm_input = tf.reshape(inputs,[-1,self.lstm_dims])
                hidden = tf.tanh(tf.nn.xw_plus_b(lstm_input,W,b))

            with tf.variable_scope('logits'):
                W = tf.get_variable('W',dtype=tf.float32,shape=[self.lstm_dim,self.num_tags],initializer=self.initializer)
                b = tf.get_variable('b',dtype=tf.float32,shape=[self.num_tags],initializer=self.initializer)

                logit = tf.nn.xw_plus_b(hidden,W,b)

            return tf.reshape(logit,[-1,self.num_steps,self.num_tags])