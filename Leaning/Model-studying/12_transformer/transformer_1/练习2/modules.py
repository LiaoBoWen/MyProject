import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers

def get_token_embedding(vocab_size,num_unit,zero_padding=True,scope='get_token_embedding'):
    with tf.variable_scope(scope):
        embedding = tf.get_variable('shared_embedding',
                                    shape=(vocab_size,num_unit),
                                    dtype=tf.float32,
                                    initializer=layers.xavier_initializer())
        if zero_padding:
            embedding = tf.concat((tf.zeros((1,num_unit)),embedding[1:,:]))
    return embedding

def position_encode(inputs,maxlen,mask,scope='position_encode'):
    E = inputs.get_shape().to_list()[-1]
    V, M = tf.shape(inputs)[0], tf.shape(inputs)[1]
    with tf.variable_scope(scope):
        position_ind = tf.tile(tf.expand_dims(tf.range(M),0),[V,1])

        position_enc = np.array([[pos / np.power(10000, (i - i % 2) / E)] for i in range(E)
                                 for pos in range(maxlen)])

        position_enc[:,0::2] = np.sin(position_enc[:,0::2])
        position_enc[:,1::2] = np.cos(position_enc[:,1::2])

        position_enc = tf.convert_to_tensor(position_enc)

        outputs = tf.nn.embedding_lookup(position_enc,position_ind)

        if mask :
            outputs = tf.where(tf.equal(inputs,0),inputs,outputs)

        return tf.to_float(outputs)


def mask(inputs,queries,keys,type):
    padding_num = - 2 ** 32 + 1

    if type in ('key','keys','k'):
        # 使用负无穷mask
        mask = tf.sign(tf.reduce_sum(keys,axis=-1))
        mask = tf.expand_dims(mask,1)
        mask = tf.tile(mask,[1,tf.shape(queries)[1],1])

        padding = tf.ones_like(mask) * padding_num
        outputs = tf.where(tf.equal(mask,0),padding,inputs)

    if type in ('qyeries','query','q'):
        mask = tf.sign(tf.reduce_sum(queries,axis=-1))
        mask = tf.expand_dims(mask,2)
        mask = tf.tile(mask,[1,1,tf.shape(keys)[1]])

        outputs = inputs * mask

    if type in ('future','right','f'):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        mask = tf.tile(tf.expand_dims(tril,0),[tf.shape(inputs),1,1])

        padding = tf.ones_like * padding_num
        outputs = tf.where(tf.equal(padding,0),padding_num,mask)

    return outputs

def attention(Q,K,V,dropout_rate,training=True,causality=False,scope='attention'):
    with tf.variable_scope(scope):
        d_model = Q.get_shape().to_list()[-1]

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))
        outputs /= d_model ** .5

        outputs = mask(outputs,Q,K,type='keys')

        if causality :
            outputs = mask(outputs,Q,K,'future')

        outputs = tf.nn.softmax(outputs)

        outputs = mask(outputs,Q,K,type='queries')

        outputs = tf.layers.dropout(outputs,dropout_rate,training=training)

        outputs = tf.matmul(outputs,V)

    return outputs

def multi_attention(queries,keys,values,dropout_rate,num_heads,training=True,causality=False,scope='multi_attention'):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        d_model = queries.get_shape().to_list()[-1]

        Q = tf.layers.dense(queries,d_model)
        K = tf.layers.dense(keys,d_model)
        V = tf.layers.dense(values,d_model)

        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),0)
        K_ = tf.concat(tf.split(K,num_heads,axis=2),0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),0)

        outputs = attention(Q_,K_,V_,dropout_rate=dropout_rate,training=training,causality=causality)

        outputs = tf.concat(tf.split(outputs,axis=0),2)

        outputs += queries

        return outputs


def layer_norm(inputs,epsilon=1e-8,scope='layer_norm'):
    with tf.variable_scope(scope):
        input_shape = inputs.get_shape()
        params = input_shape[-1:]

        mean, var = tf.nn.moments(inputs,[-1],keep_dims=True)
        outputs = (inputs - mean) / (var + epsilon) ** 0.5

        gamma = tf.get_variable('gamma',params,initializer=tf.ones_initializer())
        beta = tf.get_variable('beta',params,initializer=tf.zeros_initializer())

        outputs = gamma * outputs + beta

        return outputs

def label_smoothing(labels,epsilon=0.1):
    d_model = labels.get_shape().to_list()[-1]
    return (1 - epsilon) * labels + epsilon / d_model


def noam_sheme(lr,global_step,warmup_step):
    step = tf.cast(global_step + 1,dtype=tf.float32)
    return (lr * warmup_step ** 0.5 * tf.minimum(step * warmup_step ** -1.5, step ** -0.5))