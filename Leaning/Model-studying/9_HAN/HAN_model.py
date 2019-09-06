import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

class HAN:
    def __init__(self,max_seq_len,max_sent_len,num_classes,vocab_size,embedding_size,max_grad_norm,dropout_keep_proba,learning_rate):
        # Paraments
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        self.word_encoder_num_hidden = max_seq_len
        self.word_output_size = max_seq_len
        self.sentence_encoder_num_hidden = max_sent_len
        self.sentence_output_size =max_sent_len
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_proba = dropout_keep_proba

        # input
        # [document x sentence x word]
        self.input_x = tf.placeholder(shape=[None,None,None],
                                      dtype=tf.int32,
                                      name='input_x')
        self.input_y = tf.placeholder(shape=[None,self.num_classes],
                                      dtype=tf.int32,
                                      name='input_y')
        # [doocument x sentence]
        self.word_lengths = tf.placeholder(shape=[None,None],
                                           dtype=tf.int32,
                                           name='word_lengths')
        # [document]
        self.sentence_lengths = tf.placeholder(shape=[None,],
                                               dtype=tf.int32,
                                               name='sentence_lengths')
        # [document]
        self.is_training = tf.placeholder(dtype=tf.bool,
                                          name='is_training')

        # input_x dims
        (self.document_size,self.sentence_size,self.word_size) = tf.unstack(tf.shape(self.input_x))

        with tf.device('/gpu:0'),tf.name_scope('embedding_layer'):
            w = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0),
                            dtype=tf.float32,
                            name='w')
            self.input_x_embedded = tf.nn.embedding_lookup(w,self.input_x)

        # reshape intput_x after embedding todo 这是干嘛？
        self.input_x_embedded = tf.reshape(self.input_x_embedded,
                                           [self.document_size * self.sentence_size, self.word_size, self.embedding_size])
        self.input_x_embedded_length = tf.reshape(self.word_lengths,[self.document_size * self.sentence_size])

        with tf.variable_scope('word_level'):
            self.word_encoder_outputs = self.bidirectional_RNN(num_hidden=self.word_encoder_num_hidden,
                                                               inputs=self.input_x_embedded)
            word_level_output = self.attention(inputs=self.word_encoder_outputs,
                                               output_size=self.word_output_size)
            with tf.variable_scope('dropout'):
                print('self.is_training:{}'.format(self.is_training))
                word_level_output = layers.dropout(word_level_output,
                                                   keep_prob=self.dropout_keep_proba,
                                                   is_training=self.is_training)
        # reshape word_level output
        self.sentence_encoder_inputs = tf.reshape(word_level_output,
                                                  [self.document_size,self.sentence_size,self.word_output_size])
        with tf.variable_scope('sentence_level'):
            self.sentence_encoder_outputs = self.bidirectional_RNN(num_hidden=self.sentence_encoder_num_hidden,
                                                                   inputs=self.sentence_encoder_inputs)
            sentence_level_output = self.attention(inputs=self.sentence_encoder_outputs,
                                                   output_size=self.sentence_output_size)
            with tf.variable_scope('dropout'):
                sentence_level_output = layers.dropout(sentence_level_output,
                                                       keep_prob=self.dropout_keep_proba,
                                                       is_training=self.is_training)
        # Final model prediction
        with tf.variable_scope('classifier_output'):
            self.logits = layers.fully_connected(sentence_level_output,
                                                 self.num_classes,
                                                 activation_fn=None)
            self.predictions = tf.argmax(self.logits,axis=1,name='predictions')

        # Calculate mean cross-entropy loss
        with tf.variable_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(lables=self.input_y,logits=self.logits)
            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar('Loss',self.loss)

        # Accuracy
        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')
            tf.summary.scalar('Accuracy',self.accuracy)

    def bidirectional_RNN(self,num_hidden,inputs):
        with tf.name_scope('bidirectional_RNN'):
            encoder_fw_cell = rnn.GRUCell(num_hidden)
            encoder_bw_cell = rnn.GRUCell(num_hidden)
            ((encoder_fw_outputs,encoder_bw_outputs),(_,_)) = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell,
                                                                                              encoder_bw_cell,
                                                                                              inputs,
                                                                                              dtype=tf.float32,
                                                                                              time_major=True)
            encoder_outputs = tf.concat((encoder_fw_outputs,encoder_bw_outputs),2)  # todo 输入的维度?
            return encoder_outputs
    # end

    def attention(self,inputs,output_size):
        with tf.variable_scope('attention'):
            attention_context_vector_uw = tf.get_variable(name='attention_context_vector',
                                                          shape=[output_size],
                                                          initializer=layers.xavier_initializer(), # todo 啥初始化？
                                                          dtype=tf.float32)
            input_projection_u = layers.fully_connected(inputs,
                                                        output_size,
                                                        activation_fn=tf.tanh)
            vector_attn = tf.reduce_sum(tf.multiply(input_projection_u,attention_context_vector_uw),axis=2,keep_dims=True)      # todo shape？
            attentionn_weights = tf.nn.softmax(vector_attn,dim=1)   # todo shape？
            weighted_projection = tf.multiply(input_projection_u,attentionn_weights)    # todo shape？
            outputs = tf.reduce_sum(weighted_projection,axis=1)     # todo shape？
            return outputs
