import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

from model.basemodel import Base_model

class IDCNN_model(Base_model):
    def __init__(self,config):
        super().__init__(config)
        self.embedding = self.char_dim + self.seg_dim
        self.get_layers()
        finalOut = self.IDCNN_layer()
        self.logits = self.fc_idcnn_layer(finalOut)
        self.loss = self.loss_layer(self.logits,self.lengths)
        self.get_optimizer()


    def get_layers(self):
        self.layers = [
            {
                'dilation':1
            },
            {
                'dilation':1
            },
            {
                'dilation':2
            }
        ]

    def IDCNN_layer(self):
        model_input = tf.expand_dims(self.embedding,axis=1)
        reuse = True if self.dropout == 1.0 else False
        with tf.variable_scope('idcnn'):
            filter_weights = tf.get_variable(shape=[1, self.filter_width, self.embedding_dim, self.num_filter],
                                             initializer=self.initializer,
                                             name='idcnn_filter')

            layer_Input = tf.nn.conv2d(model_input,
                                       filter_weights,
                                       strides=[1,1,1,1],
                                       padding='SAME',
                                       name='init_layer',
                                       use_cudnn_on_gpu=True)

            finalOutFromLayers = []
            totalWidthForLastDim = 0
            layer_num = len(self.layers)
            for j in range(self.repeat_time):
                for i in range(layer_num):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == layer_num -1 else False
                    with tf.get_variable('atrous-conv-layer{}'.format(i),
                                         reuse=True if (reuse or j > 0) else False):
                        w = tf.get_variable(name='filterW',dtype=tf.float32,shape=[1,self.filter_width,self.num_filter,self.num_filter],initializer=self.initializer)
                        b = tf.get_variable(name='filterB',dtype=tf.float32,shape=[self.num_filter])

                        conv = tf.nn.atrous_conv2d(layer_Input,
                                                   w,
                                                   rate=dilation,
                                                   padding='SAME')
                        conv = tf.nn.bias_add(conv,b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter

            finalOut = tf.concat(finalOutFromLayers,axis=3)
            keepProb = 1. if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut,keepProb)
            finalOut = tf.squeeze(finalOut,[1])  # 既然有reshape，那squeeze的意义在哪里
            finalOut = tf.reshape(finalOut,[-1,totalWidthForLastDim])

            self.cnn_output_width = totalWidthForLastDim

            return finalOut

    def fc_idcnn_layer(self,cnnOut):
        with tf.variable_scope('project'):
            with tf.variable_scope('logit'):
                w = tf.get_variable('w',shape=[self.cnn_output_width,self.num_tags],initializer=self.initializer)
                b = tf.get_variable('b',initializer=tf.constant(0.001,shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(cnnOut,w,b)

                return tf.reshape(pred,[-1, self.step_num, self.num_tags])