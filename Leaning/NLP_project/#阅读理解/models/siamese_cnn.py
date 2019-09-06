import tensorflow as tf
import numpy as np


class Siamese:
    def __init__(self,config):
        self.config = config
        self.global_step = tf.train.get_or_create_global_step()
        # input_placeholder
        self.add_placeholder()
        q_embed, a_embed = self.add_embeddings()
        with tf.variable_scope('siamese',reuse=tf.AUTO_REUSE):
            self.q_trans = self.network(q_embed,reuse=False)
            self.a_trans = self.network(a_embed,reuse=True)

        self.total_loss = self.add_loss_layer(self.q_trans,self.a_trans)
        self.train_op = self.add_train_layer(self.total_loss)


    def add_placeholder(self):
        self.q = tf.placeholder(tf.int32,shape=[None,self.config.max_q_len],name='Question')
        self.a = tf.placeholder(tf.int32,shape=[None,self.config.max_a_len],name='Answer')
        self.y = tf.placeholder(tf.float32,shape=[None,],name='label')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]


    def add_embeddings(self):
        with tf.variable_scope('embeddings'):
            if self.config.embeddings is not None:
                embedding = tf.Variable(self.config.embeddings,name='embeddings',trainable=False)
            else:
                embedding = tf.get_variable('embeddings',shape=[self.config.vocab_size,self.config.embeddings],initializer=tf.uniform_unit_scaling_initializer())

            q_embed = tf.nn.embedding_lookup(embedding,self.q)
            a_embed = tf.nn.embedding_lookup(embedding,self.a)
            q_embed = tf.nn.dropout(q_embed,self.config.keep_prob)
            a_embed = tf.nn.dropout(a_embed,self.config.keep_prob)

        return q_embed, a_embed


    def network(self,x,reuse=False):
        '''核心网络'''
        conv1 = self.conv_layer(x,reuse=reuse)
        fc1 = self.fc_layer(conv1,self.config.n_weight,'fc1')
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1,self.config.n_weight,'fc2')

        return fc2


    def fc_layer(self,bottom,n_weight,name):
        assert len(bottom.get_shape()) == 2
        pre_weight = bottom.get_shape()[1]

        W = tf.get_variable(name + 'W',shape=[pre_weight,n_weight],initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name + 'b',initializer=tf.constant(0.01,dtype=tf.float32,shape=[n_weight]))
        out = tf.nn.xw_plus_b(bottom,W,b)

        return out


    def conv_layer(self,x,reuse):
        pool = list()
        max_len = x.get_shape()[1]
        x = tf.expand_dims(x,-1)
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                conv1_W = tf.get_variable('conv_W',shape=[filter_size,self.config.embedding_size,1,self.config.num_filters],initializer=tf.truncated_normal_initializer(stddev=0.01))
                conv1_b = tf.get_variable('conv_b',initializer=tf.constant(0.0,shape=[self.config.num_filters]))

                pool_b = tf.get_variable('pool_b',initializer=tf.constant(0.0,shape=[self.config.num_filters]))
                # 卷积、激活
                out = tf.nn.relu((tf.nn.conv2d(x,conv1_W,[1,1,1,1],padding='VALID') + conv1_b))
                # 池化
                out = tf.nn.max_pool(out,[1,max_len - filter_size + 1, 1, 1],[1, 1, 1, 1],padding='VALID')
                # 激活
                out = tf.nn.tanh(out + pool_b)

                pool.append(out)

                # 加入正则项
                if not reuse:       # todo what??
                    tf.add_to_collection('total_loss', 0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(conv1_W))

        total_channels = len(self.config.filter_sizes) * self.config.num_filters
        real_pool = tf.reshape(tf.concat(pool,3),[self.batch_size,total_channels])

        return real_pool

    def practic(self,x,reuse):
        pool = list()
        max_len = x.get_shape()[1]
        x = tf.expand_dims(x,-1)
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(i)):
                conv1_W = tf.get_variable('conv_W',shape=[filter_size,self.config.embedding_size,1,self.config.num_filters],initializer=tf.truncated_normal_initializer(stddev=0.01))
                conv1_b = tf.get_variable('conv_b',initializer=tf.constant(0.0,shape=[self.config.num_filters]))
                pool_b = tf.get_variable('pool_b',initializer=tf.constant(0.0,shape=[self.config.num_filters]))

                out = tf.nn.relu((tf.nn.conv2d(x,conv1_W,[1,1,1,1],padding='VALID') + conv1_b))
                out = tf.nn.max_pool(out,[1,max_len - filter_size + 1,1,1,],[1,1,1,1],padding="VALID")
                out = tf.nn.tanh(out + pool_b)

                pool.append(out)

                if reuse:
                    tf.add_to_collection('total_loss',0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(conv1_W))

        total_channels = len(self.config.filter_sizes) * self.config.num_filters
        real_pool = tf.reshape(tf.concat(pool,3),[self.batch_size,total_channels])

        return real_pool


    def add_loss_layer(self,q_trans,a_trans):
        norm1 = tf.nn.l2_normalize(q_trans,dim=1)
        norm2 = tf.nn.l2_normalize(a_trans,dim=1)

        # todo 这里的计算方法有错误
        q_a_cosine = tf.reduce_sum(tf.multiply(norm1,norm2),axis=1)

        total_loss = self.contrastive_loss(q_a_cosine,self.y)
        tf.add_to_collection('total_loss',total_loss)
        total_loss = tf.get_collection('total_loss')
        total_loss = tf.add_n(total_loss)

        return total_loss


    def contrastive_loss(self,Ew,y):
        l1 = self.config.pos_weight * tf.square(1 - Ew)
        l0 = tf.square(tf.maximum(Ew,0))

        result = tf.reduce_mean(l1 * y + (1 - y) * l0)

        return result


    def add_train_layer(self,loss):
        with tf.variable_scope('train_op'):
            op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            train_op = op.minimize(loss,global_step=self.global_step)
            return train_op