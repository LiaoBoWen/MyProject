import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.crf import crf_log_likelihood   # 这个库里面好多的crf模块


class Base_model:
    def __init__(self,config):
        self.char_dim = config.char_dim
        self.lstm_dim = config.lstm_dim
        self.seg_dim = config.seg_dim       # 切词的维度 b、i、o、e 这四个
        self.num_tags = config.num_tags     # todo 这是什么的参数
        self.num_chars = config.num_chars
        self.steps_check = config.steps_check
        self.num_segs = config.num_segs
        self.filter_width = config.filter_width  # 卷积核的大小
        self.repeat_times = config.repeat_times  # 使用膨胀卷积的吃书
        self.optimizer = config.optimizer
        self.clip = config.clip
        self.learn_rate = config.learn_rate
        self.loss = ''
        self.cnn_output_width = 0   # 输出的宽度为0
        self.initilizer = layers.xavier_initializer()
        self.get_placeholder()
        self.embedding = self.embedding(self.input_chars,self.input_segs)


    def get_placeholder(self):
        self.global_step = tf.train.get_or_create_global_step()

        self.input_chars = tf.placeholder(dtype=tf.int32,shape=[None,None],name='InputChars')
        self.input_segs = tf.placeholder(dtype=tf.int32,shape=[None,None],name='InputSegs')
        self.input_targets = tf.placeholder(dtype=tf.int32,shape=[None,None],name='InputTarget')
        self.dropout_rate = tf.placeholder(dtype=tf.float32,name='Dropout')

        #
        used = tf.sign(tf.abs(self.input_chars))
        self.length = tf.reduce_sum(used,reduction_indices=1) # todo 把这个参数换成axis试试
        self.batch_size = tf.shape(self.input_chars)[0]
        self.num_steps = tf.shape(self.input_chars)[-1]

    def embedding_layer(self,char_inputs,seg_inputs):
        '''seg_input是嵌入的分词的信息'''
        embedding = []

        with tf.variable_scope('char_embedding'):
            self.char_embedding = tf.get_variable(shape=[self.num_chars,self.char_dim],dtype=tf.float32,initializer=layers.xavier_initializer()) # 正态分布的初始方法
            embedding.append(tf.nn.embedding_lookup(self.char_dim,char_inputs))

            if self.seg_dim:  # todo 这不是提前给出了实体词性吗
                self.seg_embedding = tf.get_variable(shape=[self.num_segs,self.seg_dim],dtype=tf.float32,initializer=layers.xavier_initializer())
                embedding.append(tf.nn.embedding_lookup(self.seg_embedding,seg_inputs))
        embed = tf.concat(embedding,axis=-1)

        return embed

    def loss_layer(self,project_logits,lengths):
        '''todo 这部分的状态转移矩阵'''
        with tf.variable_scope('crf_loss'):
            small = -1024  # todo 定义看状态，看特征
            start_logits = tf.concat([small * tf.ones[self.batch_size, 1, self.num_segs], tf.zeros([self.batch_size, 1, 1])], axis=-1)
            pad_logits = small * tf.ones([self.batch_size,self.num_steps,1])
            logits = tf.concat([project_logits,pad_logits],axis=-1)
            logits = tf.concat([start_logits,logits],axis=1)
            targets = tf.concat([tf.to_int32(self.num_segs * tf.ones([self.batch_size, 1])), self.input_targets],axis=-1)

            self.trans = tf.get_variable(shape=[self.num_tags + 1, self.num_tags + 1],dtype=tf.float32,name='transition',initializer=layers.xavier_initializer())   # 特征转移矩阵


            log_likelihood, self.trans = crf_log_likelihood(inputs=self.input_chars,
                                                           tag_indices=self.input_targets,
                                                           transition_params=self.trans,
                                                           sequence_lengths=lengths + 1)
        return tf.reduce_sum(- log_likelihood)



    def get_optimizer(self):
        self.lr = tf.train.exponential_decay(self.learn_rate,self.global_step,15000,0.99,staircase=True)
        self.opt = tf.train.AdamOptimizer(self.lr)
        graph_vars = self.opt.compute_gradients(self.loss)
        clipped_grads_vars = [[tf.clip_by_value(g,-self.clip,self.clip), v] for g, v in graph_vars]
        self.train_op = self.opt.apply_gradients(clipped_grads_vars,global_step=self.global_step)
