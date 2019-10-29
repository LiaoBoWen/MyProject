import tensorflow as tf
from tensorflow.contrib import rnn

class TextRCNN:
    def __init__(self,config):
        self.sequence_length = config['sequence_length']
        self.num_classes =config['num_classes']
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.device = config['device']
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.text_hidden_size = config['text_hidden_size']

        # add placeholder
        self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,self.num_classes],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        # l2_loss
        l2_loss = tf.constant(.0)

        # embedding layer
        with tf.device(self.device),tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0),
                name='W'
            )
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)

        # Bi-LSTM layer
        with tf.name_scope('Bi-LSTM'):
            # todo rnn_hidden_size不是embedding_size吗?
            fw_cell = rnn.BasicLSTMCell(self.rnn_hidden_size)
            bw_cell = rnn.BasicLSTMCell(self.rnn_hidden_size)

            fw_cell = rnn.DropoutWrapper(fw_cell,self.dropout_keep_prob)
            bw_cell = rnn.DropoutWrapper(bw_cell,self.dropout_keep_prob)

            # todo 不好，修改
            (self.output_fw, self.output_bw), state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,inputs=self.embedded_chars,dtype=tf.float32)

        # context
        with tf.name_scope('context'):
            shape = [tf.shape(self.output_fw)[0],1,tf.shape(self.output_fw)[2]]
            self.c_left = tf.concat([tf.zeros(shape),self.output_fw[:,:-1]],axis=1,name='context_left')
            self.c_right = tf.concat([self.output_bw[:,1:],tf.zeros(shape)],axis=1,name='context_rigth')

        # word representation
        with tf.name_scope('word-representation'):
            self.x = tf.concat([self.c_left,self.embedded_chars,self.c_right],axis=2,name='x')
            embedding_size = 2 * self.rnn_hidden_size + self.embedding_size

        # text representation
        with tf.name_scope('text_representation'):
            W2 = tf.Variable(tf.random_uniform([embedding_size,self.text_hidden_size],-1.0,1.),name='W2')
            b2 = tf.Variable(tf.constant(0.1,shape=[self.text_hidden_size]),name='b2')
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik',self.x,W2) + b2)

        # max pooling
        with tf.name_scope('max_pooling'):
            self.y3 = tf.reduce_max(self.y2,axis=1)

        # final scores and predictions
        with tf.name_scope('output'):
            W4 = tf.get_variable('W4',shape=[self.text_hidden_size,self.num_classes],
            initializer = tf.contrib.layers.xavier_initializer()
            )
            b4 = tf.Variable(tf.constant(0.1,shape=[self.num_classes]),name='b4')
            l2_loss += tf.nn.l2_loss(W4)
            l2_loss += tf.nn.l2_loss(b4)   
            self.scores = tf.nn.xw_plus_b(self.y3,W4,b4,name='scores')
            self.predictions = tf.argmax(self.scores,1,name='predictions')

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32),name='accuracy')
