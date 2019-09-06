import tensorflow as tf
from tensorflow.contrib import rnn


class BiLSTM:
    def __init__(self,batch_size,max_sequence_len,embedding,embedding_size,rnn_size,margin):
        self.batch_size = batch_size
        self.max_sequence_len = max_sequence_len
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.margin = margin

        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
        self.inputQuestions = tf.placeholder(tf.int32,shape=[None,self.max_sequence_len])
        self.inputFalseAnswers = tf.placeholder(tf.int32,shape=[None,self.max_sequence_len])
        self.inputTrueAnswers = tf.placeholder(tf.int32,shape=[None,self.max_sequence_len])
        self.inputTestQuestions = tf.placeholder(tf.int32,shape=[None,self.max_sequence_len])
        self.inputTestAnswers = tf.placeholder(tf.int32,shape=[None,self.max_sequence_len])

        # embedding layer
        with tf.device('/cpu:0'),tf.name_scope('embedding_layer'):
            tf_embedding = tf.Variable(tf.to_float(self.embedding),trainable=True,name='W')
            questions = tf.nn.embedding_lookup(tf_embedding,self.inputQuestions)
            true_answers = tf.nn.embedding_lookup(tf_embedding,self.inputTrueAnswers)
            false_answers = tf.nn.embedding_lookup(tf_embedding,self.inputFalseAnswers)

            test_questions = tf.nn.embedding_lookup(tf_embedding,self.inputTestQuestions)
            test_answers = tf.nn.embedding_lookup(tf_embedding,self.inputTestAnswers)

        # LSTM
        with tf.variable_scope('LSTM_scope',reuse=None):
            question1 = self.biLSTMCell(questions,self.rnn_size)
            question2 = tf.nn.tanh(self.max_pooling(question1))
        with tf.variable_scope('LSTM_scope',reuse=True):
            true_answer1 = self.biLSTMCell(true_answers,self.rnn_size)
            true_answer2 = tf.nn.tanh(self.max_pooling(true_answer1))
            false_answer1 = self.biLSTMCell(false_answers,self.rnn_size)
            false_answer2 = tf.tanh(self.max_pooling(false_answer1))

        self.trueCosSim = self.get_cosine_similar(question2,true_answer2)
        self.falseCosSim = self.get_cosine_similar(question2,false_answer2)
        self.loss = self.get_loss(self.trueCosSim,self.falseCosSim)

        self.result = self.get_cosine_similar(test_questions,test_answers)

    def biLSTMCell(self,x,hidden_size):
        input_x = tf.transpose(x,[1,0,2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)

        lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)
        lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,input_keep_prob=self.dropout_keep_prob,output_keep_prob=self.dropout_keep_prob)

        output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,input_x,dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output)
        return output

    @staticmethod
    def get_cosine_similar(q,a):
        q1 = tf.sqrt(tf.reduce_sum(tf.multiply(q,q),1))
        a1 = tf.sqrt(tf.reduce_sum(tf.multiply(a,a),1))
        mul = tf.reduce_sum(tf.multiply(q,a),1)
        cosSim = tf.reduce_sum(tf.multiply(q1,a1),1)
        return cosSim

    @staticmethod
    def max_pooling(lstm_out):
        height = int(lstm_out.get_shape()[1])
        width = int(lstm_out.get_shape()[2])
        lstm_out = tf.expand_dims(lstm_out,-1)
        output = tf.nn.max_pool(lstm_out,ksize=[1,height,1,1],strides=[1,1,1,1],padding='VALID')
        output = tf.reshape(output,[-1,width])
        return output

    @staticmethod
    def get_loss(trueCosSim,falseCosSim,margin):
        zero = tf.tile(tf.shape(trueCosSim),margin)
        tfMargin = tf.fill(tf.shape(trueCosSim),margin)
        with tf.name_scope('loss'):
            losses = tf.maximum(zero,tf.subtract(tfMargin,tf.subtract(trueCosSim,falseCosSim)))
            loss = tf.reduce_sum(losses)
        return loss