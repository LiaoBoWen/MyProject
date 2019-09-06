import tensorflow as tf
from hypeParams import params
import numpy as np

def layer_norm(inputs,epsilon=1e-8,scope='layer_norm'):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        input_shape = inputs.get_shape()  # (?,?,512)
        params_shape = input_shape[-1:]  # (512,)

        mean, var = tf.nn.moments(inputs,[-1],keep_dims=True)   # [?,?,1]
        beta = tf.get_variable('beta',params_shape,initializer=tf.zeros_initializer())  # [512,]
        gamma = tf.get_variable('gamma',params_shape,initializer=tf.zeros_initializer())  # [512,]
        normalized = (inputs - mean) / ((var + epsilon) ** .5)  # [?,?,512]
        outputs = gamma * normalized + beta  # [?,?,512]

    return outputs


def get_token_embeddings(vocab_size,num_units,zero_pad=True):
    # todo 疑似mask?
    with tf.variable_scope('share weights matrix'):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size,num_units),
                                     initializer=tf.contrib.layers.xavier_initializer())

        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=(1,num_units)),
                                    embeddings[1:,:]),0)
    return embeddings

def scaled_dot_product_attention(Q,K,V,
                                 causality=False,dropout_rate=0.,
                                 training=True,
                                 scope='scaled_dot_product_attention'):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]  # 这里获取的是词向量的维度

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))  # 这里的矩阵乘法需要进行转置操作

        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs,Q,K,type='key')

        if causality:
            outputs  = mask(outputs,type='future')

        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs,[0,2,1])

        tf.summary.image('attention',tf.expand_dims(attention[:1],-1))

        outputs = mask(outputs,Q,K,type='query')

        outputs = tf.layers.dropout(outputs,rate=dropout_rate,training=training)

        outputs = tf.matmul(outputs,V)

    return outputs



def mask(inputs,queries=None,keys=None,type=None):
    padding_num = -2 ** 32 - 1  # 负无穷大

    if type in ('k','key','keys'):
        masks = tf.sign(tf.reduce_mean(tf.abs(keys),axis=-1))  # [N, T_k]
        masks = tf.expand_dims(masks,1)  # [N, 1, T_k]
        masks = tf.tile(masks,[1,tf.shape(queries)[1],1])   # [N, T_q, T_k]

        paddings = tf.ones_like(inputs) * padding_num
        outputs = tf.where(tf.equal(masks,0),paddings,inputs)

    elif type in ('q','query','queries'):
        masks = tf.sign(tf.reduce_mean(tf.abs(queries),axis=-1))
        masks = tf.expand_dims(masks,-1)
        masks = tf.tile(masks,[1,1,tf.shape(keys)[1]])

        outputs = inputs * masks

    elif type in ('f','future','right'):
        diag_vals = tf.ones_like(inputs[0,:,:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # todo ??
        masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(inputs)[0],1,1])

        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks,0),paddings,inputs)

    else:
        print('Check your type correctly!')
    return outputs

def multihead_attention(queries,keys,values,  # [N,T_k,d_model]
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope='multihead_attention'):
    d_model = queries.get_shape().as_list()[-1]

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        # 1. 线性变换得到Q、K、V
        Q = tf.layers.dense(queries,d_model)        # [N, T_q, d_model]
        K = tf.layers.dense(keys,d_model)
        V = tf.layers.dense(values,d_model)


        Q_ = tf.concat(tf.split(Q,num_heads,axis=2),axis=0)       # [N*h, T_q, d_model/h]
        K_ = tf.concat(tf.split(K,num_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,num_heads,axis=2),axis=0)

        # Attention
        outputs = scaled_dot_product_attention(Q_,K_,V_,causality,dropout_rate,training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs,num_heads,axis=0),axis=2)      # 将多个attention-head头拼接

        # Residual connection
        outputs += queries

        outputs = layer_norm(outputs)

    return outputs

def feed_forward(inputs,num_units,scope='positionwise_feedforward'):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        # Input layer
        outputs = tf.layers.dense(inputs,num_units[0],activation=tf.nn.relu)

        # Output layer
        outputs = tf.layers.dense(outputs,num_units[1])

        # Residual connection
        outputs += inputs

        # Layer norm
        outputs = layer_norm(outputs)

    return outputs

def label_smoothing(inputs,epsilon):
    # todo what ??
    V =  inputs.get_shape().to_list()[-1]
    return ((1 - epsilon) * inputs) + epsilon / V

def position_encoding(inputs,maxlen,masking=True,scope='position_encoding'):
    E = inputs.get_shape().to_list()[-1]
    N, T = tf.shape(inputs)[0],tf.shape(inputs)[1]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T),0),[N,1])

        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) /E) for i in range(E)]
            for pos in range(maxlen)
        ])

        # 这是什么操作
        position_enc[:,0::2] = np.sin(position_enc[:,0::2])
        position_enc[:,1::2] = np.cos(position_enc[:,1::2])

        # lokup
        outputs = tf.nn.embedding_lookup(position_enc,position_ind)

        if masking:
            outputs = tf.where(tf.equal(inputs,0),inputs,outputs)

        return tf.to_float(outputs)

def noam_scheme(init_lr,global_step,warmup_steps=4000.):
        # todo noam scheme learning rate decay????
    step = tf.cast(global_step + 1,dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)