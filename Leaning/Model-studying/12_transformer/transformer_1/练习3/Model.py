from Modules import load_vocab
from utils import convert_id_to_token
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class Transformer:
    def __init__(self,hp):
        self.hp = hp
        self.token2ind, self.idx2token = load_vocab(hp.vocab_path)
        self.embeddings = self.get_token_embeddings()


    def get_token_embeddings(self,vocab_size,num_units,zero_pad=True):
        with tf.variable_scope('share_embeddings'):
            embeddings = tf.get_variable('weight_mats',
                                         dtype=tf.float32,
                                         shape=[vocab_size,num_units],
                                         initializer=tf.contrib.layers.xavier_initialier())

            if zero_pad:
                embeddings = tf.concat([tf.zeros([1,num_units]),embeddings[1:,:]],axis=0)

        return embeddings


    def position_embedding(self,inputs,maxlen,mask=True,scope='postion_embedding'):
        '''
        :param inputs: (N,T,E)
        :param maxlen: must be >= T
        :return:
        '''
        with tf.variable_scope(scope):
            E = inputs.get_shape().as_list()[-1]
            N, T = tf.get_shape(inputs)[:2]

            position_ind = tf.tile(tf.expand_dims(tf.range(T),0),[N,1])

            position_enc = np.array(
                [pos / np.power(10000,(i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)
            )

            position_enc[:,0::2] = np.sin(position_enc[:,0::2])
            position_enc[:,1::2] = np.cos(position_enc[:,1::2])
            position_enc = tf.convert_to_tensor(position_enc,tf.float32)


            position_embed = tf.nn.embedding_lookup(position_enc,position_ind)
            if mask:
                position_embed = tf.where(tf.equal(inputs,0),inputs,position_embed)

        return position_embed


    def multi_attention(self,queries,keys,values,
                        num_heads=8,dropout_rate=0,training=True,causality=False,scope='multi_attention'):

        dim = queries.get_shape().as_list()[-1]

        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(queries,dim,activation=tf.nn.relu)
            K = tf.layers.dense(keys,dim,activation=tf.nn.relu)
            V = tf.layers.dense(values,dim,activation=tf.nn.relu)

            Q_ = tf.concat(tf.split(Q,num_heads,axis=-1),axis=0)
            K_ = tf.concat(tf.split(K,num_heads,axis=-1),axis=0)
            V_ = tf.concat(tf.split(V,num_heads,axis=-1),axis=0)

            outputs = self.scaled_dot_project_attention(Q_,K_,V_,
                                                        dropout_rate=dropout_rate,training=training,causality=causality)

            outputs = tf.concat(tf.split(outputs,axis=0),axis=-1)

        outputs += queries

        outputs = self.layer_norm(outputs)

        return outputs



    def scaled_dot_project_attention(self,Q,K,V
                                     ,dropout_rate=0,training=True,causality=False,scope='scaled_dot_project_attention'):

        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            dim = K.get_shape().as_list()[-1]

            outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))

            outputs = outputs / dim ** 0.5

            outputs = self.mask(outputs,Q,K,type='key')

            if causality:
                outputs = self.mask(outputs,Q,K,type='future')

            outputs = tf.nn.softmax(outputs)

            outputs = tf.transpose(outputs,[0,2,1])

            outputs = tf.layers.dropout(outputs,rate=dropout_rate)

            outputs = self.mask(outputs,Q,K,type='queries')

            outputs = tf.layers.dropout(outputs,rate=dropout_rate,training=training)

            outputs = tf.matmul(outputs,V)

        return outputs


    def mask(self,inputs,Q=None,K=None,type=None):
        '''
        :param inputs: shape: [N * num_heads, T, E / num_heads]
        '''
        padding_num = - 2 ** 32 + 1

        if type in ('key','keys'):
            masks = tf.sign(tf.reduce_sum(tf.abs(K),axis=-1))  # shape: [N * num_heads, T_K]
            masks = tf.expand_dims(masks,1)  # to [N * num_head, T_Q, T_K]
            masks = tf.tile(masks,[1,tf.shape(Q)[1],1])   # shape: [N * num_head, T_Q, T_K]

            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks,0),paddings,inputs)

        elif type in ('queries','query'):
            masks = tf.sign(tf.reduce_sum(tf.abs(Q),axis=-1)) # shape:[N * num_head, T_Q]
            masks = tf.expand_dims(masks,-1)
            masks = tf.tile(masks,[1,1,tf.shape(K)[1]]) # shape: [N * num_head, T_Q, T_K]

            outputs = inputs * masks

        elif type in ('future','right'):
            diag_vals = tf.ones_like(inputs[0])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(inputs)[0],1,1])

            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks,0),paddings,inputs)

        else:
            print('Type error: not in (queries,keys,future)')

        return outputs


    def layer_norm(self,inputs,epsilon=1e-8):
        with tf.variable_scope('layer_norm',reuse=tf.AUTO_REUSE):
            params_shape = inputs.get_shape()[-1:]

            beta = tf.get_variable('beta',params_shape,initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma',params_shape,initializer=tf.ones_initializer())

            mean, var = tf.nn.moments(inputs,[-1],keep_dims=True)
            outputs = (inputs - mean) / (var + epsilon) ** 0.5

            outputs = gamma * outputs + beta

        return outputs


    def feed_forword(self,inputs,filters):
        ''''''
        with tf.variable_scope('feed_forword',reuse=tf.AUTO_REUSE):
            param1 = {'inputs':inputs,'filters':filters[0],'kernel_size':1,'activation':tf.nn.relu,'use_bias':True}

            outputs = tf.layers.conv1d(**param1)

            param2 = {'inputs':outputs,'filters':filters[1],'kernel_size':1,'use_bias':True}

            outputs = tf.layers.conv1d(**param2)

            outputs += inputs

            outputs = self.layer_norm(outputs)

        return outputs


    def noam_scheme(self,lr,global_step,warmup=4000.):
        '''
        <===example===>: todo 发现lr衰减的太快，可能导致模型收敛困难
        lr=0.00002
        step =1
        lr = lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5); step += 1; lr
        Out[147]: 5e-09
        lr = lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5); step += 1; lr
        Out[148]: 2.5e-12
        lr = lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5); step += 1; lr
        Out[149]: 1.875e-15
        lr = lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5); step += 1; lr
        Out[150]: 1.8749999999999997e-18
        lr = lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5); step += 1; lr
        Out[151]: 2.3437499999999996e-21
        lr = lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5); step += 1; lr
        Out[152]: 3.515625e-24
        '''
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return lr * warmup ** 0.5 * min(step * warmup ** -1.5, step ** -0.5)


    def label_smooth(self,y,epsilon=0.1):
        V = y.get_shape().as_list()[-1]
        return (1 - epsilon) * y + epsilon / V


    def encoder(self,xs,training=True):

        encode = tf.nn.embedding_lookup(self.embeddings,self.xs)
        # scale  缩放
        encode *= self.hp.num_units ** 0.5

        encode += self.position_embedding(encode,maxlen=self.hp.maxlen)
        encode = tf.layers.dropout(encode,self.dropout_rate,training=training)  #dropout'training params

        for i in range(self.hp.num_blocks):
            with tf.variable_scope('num_block_{}'.format(i + 1),reuse=tf.AUTO_REUSE):
                encode = self.multi_attention(encode,encode,encode,
                                              num_heads=self.hp.num_heads,causality=False,training=training)
                encode = self.feed_forword(encode,filters=[self.hp.dim_feed_forword,self.hp.num_units])

        return encode



    def decoder(self,ys,memory,training=True):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            decode = tf.nn.embedding_lookup(self.embeddings,self.decode_inputs)
            decode *= self.hp.num_units ** 0.5
            decode += self.position_embedding(decode,self.hp.maxlen)

            decode = tf.layers.dropout(decode,rate=self.dropout_rate,training=training  )

            for i in range(self.hp.num_blocks):
                with tf.variable_scope('num_block_{}'.format(i + 1),reuse=tf.AUTO_REUSE):
                    decode = self.multi_attention(decode,decode,decode,
                                                  num_heads=self.hp.num_heads,training=training,causality=True,scope='self_attention')
                    decode = self.multi_attention(decode,memory,memory,
                                                  num_heads=self.hp.num_heads,training=training,causality=False,scope='vanilla_attention')
                    decode = self.feed_forword(decode,[self.hp.dim_feed_forword,self.hp.num_units])

            logit = tf.layers.dense(decode,len(self.token2ind))
            y_hat = tf.to_int32(tf.argmax(logit,axis=-1))

            return logit, y_hat


    def train(self,xs,ys):
        memory = self.encoder()

        logit, self.y_hat = self.decoder(memory)

        y_ = self.label_smooth(tf.one_hot(self.ys,depth=self.hp.vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y_)
        nopadding = tf.to_float(tf.not_equal(self.ys,self.token2ind['<PAD>']))
        self.loss = tf.reduce_sum(loss * nopadding) / (tf.reduce_sum(nopadding) + 1e-7)

        self.global_step = tf.train.get_or_create_global_step()
        lr = self.noam_scheme(self.hp.lr,self.global_step,self.hp.warmup_step)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step)



        tf.summary.scalar('learning_rate',lr)
        tf.summary.scalar('loss',self.loss)

        self.summary = tf.summary.merge_all()



    def eval(self,xs,ys):
        self.decode_inputs = tf.ones((tf.shape(self.xs)[0],1),tf.int32) * self.token2ind['<S>']
        temp_decode = self.decode_inputs
        memory = self.encoder(training=False)

        # generate
        for _ in range(10):
            logit, self.y_hat = self.decoder(memory,training=False)
            self.decode_inputs = tf.concat([temp_decode, self.y_hat],1)           # 每次循环一次都会会增加一个字


        # pred = []
        # for y in y_hat:
        #     pred.append(convert_id_to_token(y,self.idx2token))
        #
        # return y_hat
