import numpy as np
import tensorflow as tf

class Transformer:
    def __init__(self,config,wordEmbedding):
        self.inputs = tf.placeholder(tf.int32,[None,config.seqLen],name='inputs')
        self.outputs = tf.placeholder(tf.float32,[None,1],name='outputs')

        self.dropoutKeepProb = tf.placeholder(tf.float32,name='dropoutKeepPorb')

        # todo 虽然原来论文里面的位置编码是使用的相对编码，
        #  但是这里使用的one-hot编码的效果更好,可能是因为使用的是小数据
        self.positionEmbedding = tf.placeholder(tf.float32,[config.seqLen,config.seqLen],name='positionEmbedding')

        self.config = config

        l2Loss = tf.constant(0.0)

        with tf.variable_scope('embedding'):
            wordEmbedding = tf.Variable(tf.cast(wordEmbedding,dtype=tf.float32),name='wordEmbedding')

            self.embedded = tf.nn.embedding_lookup(wordEmbedding,self.inputs,name='embedded')
            self.embeddedWords = tf.concat([self.embedded,self.inputs],axis=-1)


        with tf.variable_scope('transformer'):
            for i in range(config.numBlocks):
                with tf.variable_scope('block-{}'.format(i)):
                    # todo 难道这里的queries只有位置编码吗？！  还有norm和自注意后的残差怎么不见了
                    multiHeadAtt = self.multiHeadAttention(self,rawKeys=self.inputs,queries=self.positionEmbedding,keys=self.embeddedWords,
                                                           numUnits=None,causality=False,scope='multiheadAttention')

                    self.embeddingWords = self.feedForword(multiHeadAtt,[self.config.model,self.config.model.embeddingSize + self.config.seqLen])
            outputs = tf.reshape(self.embeddingWords,[-1,config.seqLen * (config.model.embeddingSize + config.seqLen)])

        outputSize = outputs.get_shape()[-1].value

        with tf.name_scope('dropout'):
            outputs = tf.nnn.dropout(outputs,keep_prob==self.dropoutKeepProb)
        with tf.name_scope('output'):
            outputW = tf.get_variable(
                'outputW',shape=[outputSize,1],
                initializer=tf.contrib.layer.xavier_initializer()
            )
            outputB = tf.Variable(tf.constant(0.1,shape=[1],name='outputB'))
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(outputs,outputW,outputB)
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions,0.0),tf.float32,name='binaryPreds')

        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions,labels=self.outputs)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def multiHeadAttention(self,rawKeys,queries,keys,numUnits=None,causality=False,scope='multiHeadAttention'):
        if numUnits is None:
            numUnits = queries.get_shape().as_list()[-1]

        # linear 变换， 还有一个作用：这里的 queries和keys的最后一维的长度不一样，所以，可以把维度dense到相同的长度
        Q = tf.layers.dense(queries,numUnits,activation=tf.nn.relu)
        K = tf.layers.dense(keys,numUnits,activation=tf.nn.relu)
        V = tf.layers.dense(keys,numUnits,activation=tf.nn.relu)

        # real multi-heads
        Q_ = tf.concat(tf.split(Q,self.config.numHeads,axis=-1),axis=0)
        K_ = tf.concat(tf.split(K,self.config.numHeads,axis=-1),axis=0)
        V_ = tf.concat(tf.split(V,self.config.numHeads,axis=-1),axis=0)

        q_k_score = tf.matmul(Q_,tf.transpose(K_,[0,2,1]))

        devided = q_k_score / (K.get_shape().as_list()[-1])

        # 由于多头的原因，第一维发生了变化，变为了原来的numHead倍
        keysmask = tf.tile(rawKeys,[self.config.numHeads])

        keysmask = tf.tile(tf.expand_dims(keysmask,1),[1,tf.shape(queries)[1],1])

        paddings = tf.ones_like(keysmask) * (-2 ** 32 + 1)

        maskedSim = tf.where(tf.equal(keysmask,0),paddings,devided)

        # 由于这个是分类任务，所以不需要mask将来的部分，但是还是写一下
        if causality:
            diagVals = tf.ones_like(maskedSim[0,:,:])
            tril = tf.contrib.linalg.LinearOperatoe(diagVals).to_dense()
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(maskedSim)[0],1,1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            maskedSim = tf.where(tf.equal(paddings,0),paddings,maskedSim)

        weights = tf.nn.softmax(maskedSim)

        results = tf.matmul(weights,V_)

        # 还原成原来的shape
        results = tf.concat(tf.split(results,axis=0),axis=-1)

        results = tf.nn.dropout(results,self.dropoutKeepProb)

        return results


    def feedForword(self,inputs,filters,scope='multiHeadAtt'):
        param1 = {'inputs':inputs,'filters':filters[0],'kernel_size':1,'activate':None}

        outputs = tf.layers.conv1d(**param1)

        param2 = {'inputs':outputs,'filters':filters[1],'kernel_size':1,'activate':None}

        outputs = tf.layers.conv1d(**param2)

        outputs += inputs

        outputs = self.layer_norm(outputs)

        return outputs



    def layer_norm(self,inputs,epsilon=1e-8,scope='layer_norm'):
        shape = inputs.get_shape().as_lsit()[-1:]

        mean, var = tf.nn.moments(inputs,axes=-1,keep_dims=True)

        gamma = tf.Variable(tf.ones(shape))
        beta = tf.Variabel(tf.zeros(shape))

        normlized = (inputs - mean) / (var + epsilon) ** 0.5

        outputs =  gamma * normlized + beta

        return outputs

    def positionEmbedding(self,inputs,maxlen,mask=True,scope='positionEmbedding'):
        '''
        :param inputs:  3-dim tensor:(N,T,E)
        :param maxlen: must be >= T
        :param mask: if it's True, set the place of padding to zeros

        TODO note:在第一列concat一列，使得inputs‘shape == (N,T+1,E)，因为第一列的indice为0，会被作为padding而错误mask掉
        '''
        E = inputs.get_shape().as_list()[-1]  # 静态
        N, T = tf.shape(inputs)[:2]

        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            position_ind = tf.tile(tf.expand_dims(tf.range(T),axis=0),[N,1])

            position_enc = np.array(
                [pos / np.power(10000,(i - i % 2) / E) for i in range(E)]
                for pos in range(maxlen)
            )
            position_enc[:,0::2] = np.sin(position_enc[:,0::2])
            position_enc[:,1::2] = np.cos(position_enc[:,1::2])

            position_embed = tf.nn.embedding_lookup(position_enc,position_ind) # todo 注意：由于padding是0，第一个字或者词的indice也是0，所以第一个字会被mask掉，在计算position之前需要队数据做一下处理，比如向右移动一格(相当于子啊一格位置conct一列)

            if mask:
                position_embed = tf.where(tf.equal(inputs,0),position_embed)

            return position_embed

