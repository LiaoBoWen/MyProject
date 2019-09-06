import tensorflow as tf

class SiameseNN:
    def __init__(self,config):
        self.config = config
        self.global_step = tf.train.get_or_create_global_step()

        # input
        self.add_placeholders()

        q_embed, a_embed = self.add_embeddings()
        with tf.variable_scope('siamese',reuse=tf.AUTO_REUSE):
            self.q_trans = self.network(q_embed)
            self.a_trans = self.network(a_embed)

        self.add_loss_op(self.q_trans,self.a_trans)
        self.train_op = self.add_train_op(self.total_loss)



    def add_placeholders(self):
        self.q = tf.placeholder(tf.int32,shape=[None,self.config.max_q_len],name='Answer')
        self.a = tf.placeholder(tf.int32,shape=[None,self.config.max_a_len],name='Question')
        self.y = tf.placeholder(tf.float32,shape=[None,],name='label')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_porb')
        self.batch_size = tf.shape(self.q)[0]


    def add_embeddings(self):
        with tf.variable_scope('embeddings'):
            if self.config.embeddings is not None:
                self.embeddings = tf.Variable(self.config.embeddings,name='embeddings',trainable=False) # 不许训练
            else:
                self.embeddings = tf.get_variable('embeddings',shape=[self.config.vocab_size,self.config.embedding_size],initializer=tf.uniform_unit_scaling_initializer())

            q_embed = tf.nn.embedding_lookup(self.embeddings,self.q)
            a_embed = tf.nn.embedding_lookup(self.embeddings,self.a)

            q_embed = tf.nn.dropout(q_embed,keep_prob=self.keep_prob)
            a_embed = tf.nn.dropout(a_embed,keep_prob=self.keep_prob)

            return q_embed, a_embed

    def network(self,x):
        max_len = tf.shape(x)[1]
        x = tf.reshape(x,[-1,x.get_shape()[-1]])

        fc1 = self.fc_layer(x,self.config.hidden_size,'fc1')
        ac1 = tf.nn.relu(fc1)

        fc2 = self.fc_layer(ac1,self.config.hidden_size,'fc2')
        ac2 = tf.nn.relu(fc2)

        ac3 = tf.reshape(ac2,[self.batch_size,max_len,ac2.get_shape()[1]])
        ac3 = tf.reduce_mean(ac3,axis=1)

        fc3 = self.fc_layer(ac3,self.config.output_size,'fc3')

        return fc3


    def fc_layer(self,bottom,n_weight,name):
        assert len(bottom.get_shape()) == 2

        pre_weight = bottom.get_shape()[1]
        initializer = tf.truncated_normal_initializer(stddev=0.01,)

        W = tf.get_variable(name + 'W',dtype=tf.float32,shape=[pre_weight,n_weight],initializer=initializer)
        b = tf.get_variable(name + 'b',dtype=tf.float32,initializer=tf.constant(0.01,shape=[n_weight],dtype=tf.float32))

        fc = tf.nn.xw_plus_b(bottom,W,b)

        return fc


    def add_loss_op(self,q_trans,a_trans):
        # 这里是计算余弦距离？如果是，有点奇怪
        norm1 = tf.nn.l2_normalize(q_trans,dim=1)
        norm2 = tf.nn.l2_normalize(a_trans,dim=1)
        self.q_a_cosine = tf.reduce_sum(tf.multiply(norm1,norm2),1)

        loss = self.contrastive_loss(self.q_a_cosine,self.y)
        # tf.add_to_collection('total_loss',loss)
        # loss = tf.get_collection('total_loss')
        self.total_loss = tf.add_n(loss)


    def contrastive_loss(self,Ew,y):
        l1 = self.config.pos_weight * tf.square(1 - Ew)
        l0 = tf.square(tf.maximum(Ew,0))
        loss = tf.reduce_mean(y * l1 + (1 - y) * l0)

        return loss

    def add_train_op(self,loss):
        with tf.variable_scope('train'):
            op = tf.train.AdamOptimizer(loss)
            self.train_op = op.minimize(loss,self.global_step)


