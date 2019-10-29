import tensorflow as tf
# import numpy as np

class TextCNN:
    '''
    嵌入层>>卷积层>>最大池化层>>softmax层
    '''
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):
        # placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input_x')   # 因为后面的embedding_lookup所以使用的int32
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        # keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

    # 1、 构建中间层，单词转换成向量的形式
        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name='W') 
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)         
            # 由于进行了embedding_lookup，维度升高了一（None：vocab_size）
            # 增加一个维度（变成4维）[None,sequence_length,embedding_chars,1]  因为语句的话具备1个通道，类似于灰度图片，而不是rgb图片
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)

    # 2、加入卷积层，激活层，池化层
        #创建一个卷积+激活给每个filter size
        pooled_outputs = []
        for i ,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                # 卷积层
                filter_shape = [filter_size,embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
                b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b')
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                   W,
                                   strides=[1,1,1,1],
                                   padding='VALID',     # 这里必须使用VALID，如果使用的SAME会进行补0操作，没有意义
                                   name='conv') + b

                # 激活层
                h = tf.nn.relu(conv,name='relu')

                # 最大池化
                pooled = tf.nn.max_pool(h,
                                        ksize=[1,sequence_length - filter_size + 1, 1, 1],      # 这里的的seqence_length - filter_size + 1是通过计算出来的，所以在此之前需要搞清楚卷积之后的矩阵图的shape,所以这里有点可怕，如果句子的长度过长，池化层的尺寸会很大
                                        strides=[1,1,1,1],
                                        padding='VALID',
                                        name='pool')         # [None,1,1,num_filter]

                pooled_outputs.append(pooled)

        # 聚合所有的pooled feature，拉平，随后全连接
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)       # 把最后一个维度（最内部的维度）进行合并
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

    # 3、dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
    # 4、FC （output）
        # Final(unnomalized) score and predictioins
        with tf.name_scope('output'):
            W = tf.get_variable('W',
                                shape=[num_filters_total,num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())     #这个初始化器是用来保持每一层的梯度大小都差不多相同。 比如：1 2 3 4 5
            b = tf.Variable(tf.constant(0.1,shape=[num_classes],name='b'))
        
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)     
            self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name='score')  # todo ?
            self.predictions = tf.argmax(self.scores,1,name='predictioins')

    # 5、计算Loss
        # 计算cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    # 6、计算Accuracy
        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')
