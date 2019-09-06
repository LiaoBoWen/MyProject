import tensorflow as tf

from tensorflow.contrib import rnn

class Model:
    def __init__(self,batch_size,seq_len,vocab_size,unit_num,embedding_num,classes_num,learning_rate,init_embedding,is_training=True):
        self.input_x = tf.placeholder(shape=(batch_size,seq_len),name='input_x',dtype=tf.int32)
        self.input_y = tf.placeholder(shape=(batch_size,),name='input_y',dtype=tf.int32)
        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.unit_num = unit_num
        self.vocab_size = vocab_size
        self.embedding_num = embedding_num
        self.classes_num = classes_num
        self.learning_rate = learning_rate
        self.init_embedding = init_embedding
        self.is_training = is_training   # 是否训练， 预测的时候修改为False

        # 网络参数的初始化
        self.W_project = tf.get_variable(name='W_project',shape=(self.unit_num * 2,self.classes_num),initializer=tf.random_normal_initializer(stddev=0.1))
        self.b_project = tf.get_variable(name='b_project',shape=(self.classes_num),initializer=tf.random_normal_initializer(stddev=0.1))

        self.global_step = tf.train.get_or_create_global_step()    # here is differece from raw codes
        self.epoch = tf.Variable(0,dtype=tf.int32,name='epoch')
        self.epoch_increase = tf.assign(self.epoch,tf.add(self.epoch, tf.constant(1)),name='epoch_increase')



        # Inference
        self.inference()
        if self.is_training:
            return
        self.run_loss()
        self.run_train(self.loss)
        acc_pred = tf.cast(tf.equal(tf.cast(self.pred,tf.int32),self.input_y),tf.float32)
        self.acc = tf.reduce_mean(acc_pred,name='accuracy')



    def inference(self):
        with tf.variable_scope('Embedding'):
            # self.embedding_num = tf.Variable(self.init_embedding,name='embedding')
            self.embedding = tf.get_variable(name='embedding',shape=(self.vocab_size,self.embedding_num),initializer=tf.random_normal_initializer(stddev=0.1))
            self.input = tf.nn.embedding_lookup(self.embedding,self.input_x)

        with tf.variable_scope('Inference'):
            fw_lstm_cell = rnn.LSTMCell(num_units=self.unit_num)
            bw_lstm_cell = rnn.LSTMCell(num_units=self.unit_num)

            if self.dropout_rate is not None:
                fw_lstm_cell = rnn.DropoutWrapper(fw_lstm_cell)
                bw_lstm_cell = rnn.DropoutWrapper(bw_lstm_cell)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, self.input,dtype=tf.float32)
            outputs = tf.concat(outputs,axis=-1)

            cell = rnn.BasicLSTMCell(self.unit_num * 2)

            if self.dropout_rate is not None :
                cell = rnn.DropoutWrapper(cell)

            _, finish_state = tf.nn.dynamic_rnn(cell,outputs,dtype=tf.float32)
            finish_state_h = finish_state[0]


            finish_output = tf.layers.dense(finish_state_h,self.unit_num * 2,activation=tf.tanh)

            self.logits = tf.nn.xw_plus_b(finish_output,self.W_project,self.b_project,name='logits')
            self.pred = tf.argmax(self.logits,axis=1,name='pred')



    def run_loss(self,lambda_g=7e-3):
        with tf.variable_scope('Loss'):
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss_ = tf.reduce_mean(loss_)
            # loss__ = tf.add_n(tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name) * lambda_g
            self.loss = loss_


    def run_train(self,loss):
        with tf.variable_scope('Train'):
            option = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = option.minimize(loss,global_step=self.global_step)

def test_():
    X = [[2,2,2],[3,4,4]]
    Y = [0,1]
    model = Model(batch_size=2,seq_len=3,unit_num=300,vocab_size=3,embedding_num=300,classes_num=2,learning_rate=1,init_embedding=None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            feed_dict = {
                model.input_x : X,
                model.input_y : Y,
                model.dropout_rate : 0.5
            }
            step, _, loss, acc, pred ,epoch = sess.run([model.global_step,model.train_op,model.loss,model.acc,model.pred,model.epoch_increase],feed_dict=feed_dict)
            print('the epoch is:{} ,step:{} , loss:{} ,  pred:{}, acc:{}'.format(epoch, step, loss, pred, acc))
