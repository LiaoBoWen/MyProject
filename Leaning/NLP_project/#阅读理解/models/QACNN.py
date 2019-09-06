import tensorflow as tf
import numpy as np

class QACNN:
    def __init__(self,config):
        self.config = config
        # add placeholder
        self.add_placeholder()
        q_embed, aplus_embed, aminus_embed = tf.nn.embedding_lookup()
        with tf.variable_scope('siamese')


    def add_placeholder(self):
        self.q = tf.placeholder(tf.int32,shape=[None,self.config.max_q_len],name='Question')
        self.aplus = tf.placeholder(tf.int32,shape=[None,self.config.max_a_len],name='PosAns')
        self.aminus = tf.placeholder(tf.int32,shape=[None,self.config.max_a_len],name='NegAns')

        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]


    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings,name='embeddings',trainable=False)
            else:
                embeddings = tf.get_variable('embeddings',shape=[self.config.vocab_size,self.config.embeddings],initializer=tf.truncated_normal_initializer(stddev=0.01))

            q_embed = tf.nn.embedding_lookup(embeddings,self.q)
            aplus_embed = tf.nn.embedding_lookup(embeddings,self.aplus)
            aminus_embed = tf.nn.embedding_lookup(embeddings,self.aminus)
            q_embed = tf.nn.dropout(q_embed,self.keep_prob)
            aplus_embed = tf.nn.dropout(aplus_embed,self.keep_prob)
            aminus_embed = tf.nn.dropout(aminus_embed,self.keep_prob)

        return q_embed, aplus_embed, aminus_embed
