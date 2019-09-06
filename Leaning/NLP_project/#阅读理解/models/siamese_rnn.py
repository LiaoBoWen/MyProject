import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

class SiameseNN:
    def __init__(self,config):
        self.config = config
        self.global_step = tf.train.get_or_create_global_step()
        # add placeholder
        self.add_placeholder()
        q_embed, a_embed= self.add_embedding_layer()
        with tf.variable_scope('siamese',reuse=tf.AUTO_REUSE):
            self.q_trans = self.network(q_embed)
            self.a_trans = self.network(a_embed)

        self.total_loss = self.add_loss_layer(self.q_trans,self.a_trans)
        self.train_op = self.get_train_op(self.total_loss)


    def add_placeholder(self):
        self.q = tf.placeholder(tf.int32,[None,self.config.max_q_len],name='Question')
        self.a = tf.placeholder(tf.int32,[None,self.config.max_a_len],name='Answer')
        self.y = tf.placeholder(tf.float32,[None,],name='y')

        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]

    def add_embedding_layer(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings,name='embeddings',trainable=False)
            else:
                embeddings = tf.get_variable('embeddings',dtype=tf.float32,shape=[self.config.vocab_size,self.config.embeddings],initializer=tf.truncated_normal_initializer())

            q_embed = tf.nn.embedding_lookup(embeddings,self.q)
            a_embed = tf.nn.embedding_lookup(embeddings,self.a)
            # todo 这么早就dropout？！！
            q_embed = tf.nn.dropout(q_embed,keep_prob=self.config.keep_prob)
            a_embed = tf.nn.dropout(a_embed,keep_prob=self.config.keep_prob)

            return q_embed, a_embed


    def network(self,x):
        # 可加快速度
        inputs = tf.transpose(x,[1,0,2])
        # inputs = tf.reshape(inputs,[-1,self.config.embedding_size])
        # inputs = tf.split(inputs,max_len,0)
        inputs = tf.expand_dims(inputs,0)
        rnn1 = self.rnn_layer(inputs)
        fc1 = self.fc_layer(rnn1,self.config.hidden_size,'fc1')
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1,self.config.hidden_size,'fc2')
        ac2 = tf.nn.relu(fc2)

        return ac2


    def fc_layer(self,bottom,n_weight,name):
        assert len(bottom.get_shape()) == 2
        pre_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name + 'W', dtype=tf.float32,shape=[pre_weight,n_weight],initializer=initer)
        b = tf.get_variable(name + 'b', initializer=tf.constant(0.0,shape=[n_weight],dtype=tf.float32))
        fc = tf.nn.xw_plus_b(bottom,W,b)

        return fc


    def rnn_layer(self,x):
        if self.config.cell_type == 'lstm':
            birnn_fw, birnn_bw = self.bi_lstm(self.config.num_units,self.config.layer_size,self.config.keep_prob)
        else:
            birnn_fw, birnn_bw = self.bi_gru(self.config.num_units,self.config.layer_size,self.config.keep_prob)

        outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(birnn_fw,birnn_bw,x)
        output = tf.reduce_mean(outputs,0)

        return output



    def bi_lstm(self,num_units,layer_size,keep_prob):
        with tf.variable_scope('fw_rnn'):
            fw_lstm_list = [rnn.LSTMCell(num_units) for _ in layer_size]
            fw_lstm_m = rnn.DropoutWrapper(rnn.MultiRNNCell(fw_lstm_list),output_keep_prob=keep_prob)

        with tf.variable_scope('bw_rnn'):
            bw_lstm_list = [rnn.LSTMCell(num_units) for _ in layer_size]
            bw_lstm_m = rnn.DropoutWrapper(rnn.MultiRNNCell(bw_lstm_list),output_keep_prob=keep_prob)

        return fw_lstm_m, bw_lstm_m


    def bi_gru(self,num_units,layer_size,keep_prob):
        with tf.variable_scope('fw'):
            fw_gru_list = [rnn.LSTMCell(num_units) for _ in layer_size]
            fw_gru_m = rnn.DropoutWrapper(rnn.MultiRNNCell(fw_gru_list),output_keep_prob=keep_prob)
        with tf.variable_scope('bw'):
            bw_gru_list = [rnn.LSTMCell(num_units) for _ in layer_size]
            bw_gru_m = rnn.DropoutWrapper(rnn.LSTMCell(bw_gru_list),output_keep_prob=keep_prob)

        return fw_gru_m, bw_gru_m


    def add_loss_layer(self,q_trans,a_trans):
        norm1 = tf.nn.l2_normalize(q_trans)
        norm2 = tf.nn.l2_normalize(a_trans)

        q_a_cosine = tf.reduce_sum(tf.multiply(norm1,norm2),axis=1)
        total_loss = self.contrastive_loss(q_a_cosine,self.y)
        return total_loss


    def contrastive_loss(self,Ew,y):
        l1 = self.config.pos_weight * tf.square(1 - Ew)
        l0 = tf.square(tf.maximum(Ew))

        loss = tf.reduce_sum(y * l1 + (1 - y) * l0)
        return loss


    def get_train_op(self,loss):
        with tf.variable_scope('train_op'):
            op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            train_op = op.minimize(loss)
        return train_op