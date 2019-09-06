import tensorflow as tf
from tensorflow.contrib import layers


class TextCNN:
    # 这里的输入使用的是word2vec词向量
    def __init__(self,sequence_length,num_classes,embedding_size,filter_sizes,num_filters,l2_reg_lambda=.0):
        # Placeholder of input, output, dropout
        self.input_x = tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        # 追踪l2
        l2_loss = tf.constant(.0)

    # 构建中间层，单词转换成向量的形式
        #Embedding layer
        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            # # 词表向量W
            # self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.,1.),name='W')
            # self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.input_x,-1)

    # 加入卷积层，激活层，池化层
        # 创建一个卷积+激活给每个filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                # 卷积层
                filter_shape = [filter_size,embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                b = tf.Variable(tf.constant(.1,shape=[num_filters]),name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1,1,1,1],
                                    padding='VALID',    # 使用的VALID模式
                                    name='conv') + b
                # 激活层
                h = tf.nn.relu(conv,name='relu')

                # 最大池化
                pooled = tf.nn.max_pool(h,
                                        ksize=[1,sequence_length - filter_size + 1,1,1],
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        name='pool')

                pooled_outputs.append(pooled)

        # 聚合所有的pooled
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

    # Dropout 输出之前使用
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
    # FC
        with tf.name_scope('output'):
            W = tf.get_variable('W',
                                shape=[num_filters_total,num_classes],
                                initializer=layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1,shape=[num_classes]),name='b')

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name='score')
            self.predictions = tf.argmax(self.scores,1,name='predictions')

    # 计算Loss
        # 计算cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    # 计算Accuracy
        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')