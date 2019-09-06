import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class TextRNN:
    def __init__(self,num_classes,learn_rate,batch_size,decay_steps,decay_rate,sequence_length,
                 vocab_size,embed_size,is_training,initializer=tf.random_normal_initializer(stddev=0.1)):   # todo 这个是啥初始化？
        '''初始化超参数'''
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size =embed_size        # 最初的hidden_size可以设置为embed_size，然后再做调整
        self.is_training = is_training
        self.learning_rate = learn_rate
        self.initializer = initializer
        self.num_sampled = 20

        # add palceholder
        self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name='input_x')
        self.input_y =tf.placeholder(tf.int32,[None],name='input_y') # todo  因为后面的sparse_softmax_cross_entropy_with_logits方法，会进行one-hot编码所以这里直接传一维数组就行==>[None,num_classes]
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        self.global_step = tf.Variable(0,trainable=False,name='Global_Step')
        self.epoch_step = tf.Variable(0,trainable=False,name='Epoch_Step')
        self.epoch_increament = tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() # [None, self.labels_size]  todo     main!
        if not is_training:
            return
        self.loss_val = self.loss() # ==> self.loss_nec()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits,1,name='predictions') # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32),self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='Accuracy') # shape = ()

    def instantiate_weights(self):
        '''定义所有的权重'''
        with tf.name_scope('embedding'):    #embedding matrix
            self.Embedding = tf.get_variable('Embedding', shape=[self.vocab_size,self.embed_size], initializer=self.initializer) # [vocab_size,embed_size] tf.random_uniform([self.vocab_size,self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable('W_projection', shape=[self.hidden_size * 2, self.num_classes], initializer=self.initializer) # [embed_size,label_size]
            self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes], initializer=self.initializer)

    def inference(self):
        '''这是主要的计算图: 1. embedding layer 2. Bi-LSTM ==>dropout 3.LSTM layer ==>dropout 4.FC layer 5.softmax layer'''
        # 1. get embedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) # shape:[None,sentence_length,embed_size]

        # 2. Bi-LSTM layer
        # define lstm cell :get lstm cell output
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size) # forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size) # backward direction cell

        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
            # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
            #                            output: A tuple (outputs, output_states)
            #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype=tf.float32) # [batch_size,sequence_length,hidden_size] # 创建一个双向动态的RNN
        print('outputs:===>',outputs) # outputs:(<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>, <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))
        output_rnn = tf.concat(outputs,axis=2) # [batch_size,sequence_length,hidden_size * 2] todo 这个维度是time_major=False的时候，但是使用True的话更快一点。把最后（里）一个维度进行拼接，所以 *2

        # 3. second LSTM layer
        rnn_cell = rnn.BasicLSTMCell(self.hidden_size * 2)
        if self.dropout_keep_prob is not None:
            rnn_cell = rnn.DropoutWrapper(rnn_cell,output_keep_prob=self.dropout_keep_prob)
        _, final_state_c_h = tf.nn.dynamic_rnn(rnn_cell,output_rnn,dtype=tf.float32)
        final_state = final_state_c_h[1]        # todo output 包含隐层的输出，但是state的输出是（C，H）的结构,这里输出H ==>https://www.cnblogs.com/lovychen/p/9294624.html

        # 4. FC layer
        output = tf.layers.dense(final_state,self.hidden_size * 2 ,activation=tf.nn.tanh)

        # 5. logits(use linear layer)
        with tf.name_scope('output'):# inputs: a tensor of shape [batch_size,dim]
            logits = tf.matmul(output,self.W_projection) + self.b_projection # [batch_size,num_classes]
        return logits

    def loss(self,l2_lambda=1e-4):  # todo 研究一下参数的设置
        with tf.name_scope('loss'):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            '''
            sparse_softmax_cross_entropy_with_logits中 lables接受直接的数字标签 
            如[1], [2], [3], [4] （类型只能为int32，int64） 
            而softmax_cross_entropy_with_logits中 labels接受one-hot标签 
            如[1,0,0,0], [0,1,0,0],[0,0,1,0], [0,0,0,1] （类型为int32， int64）
            相当于sparse_softmax_cross_entropy_with_logits 对标签多做一个one-hot动作
            '''
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_nce(self,l2_lambda=1e-4): # 1e-4 ==> 1e-3
        '''使用一种新的损失去计算图 todo NCE_loss '''
        '''tf.nce_loss automatically draws a new sample of the negative labels each'''
        # time we evaluate the loss
        if self.is_training:
            labels = tf.expand_dims(self.input_y,-1)    # 在最后的维度加一个维度
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W_projection),
                               biases=self.b_projection,
                               labels=labels,
                               inputs=self.output_run_last,     #
                               num_sampled=self.num_sampled, # scalar,100
                               num_classes=self.num_classes,
                               partition_strategy='div'))   # scalat 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        '''based on the loss ,use SGD to update parameter'''
        # todo 使用最小loss进行修改
        # learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate,staircase=True)
        # train_op =tf.contrib.layers.optimize_losses(self.loss_val,global_step=self.global_step,learning_rate=learning_rate,optimizer='Adam')
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val,global_step=self.global_step)
        return train_op


# test start
def test():
    num_classes =10
    learning_rate = 1e-3
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training =  True
    dropout_keep_prob = 1
    textRNN = TextRNN(num_classes,learning_rate,batch_size,decay_steps,decay_rate,sequence_length,vocab_size,embed_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size,sequence_length))    # [None,self.sequence_length]
            input_y =np.array([3,0,5,1,7,2,3,0]) # np.zeros((batch_size),dtype=np.int32) # [None,self.sequence_length]
            loss, acc, predict, _ = sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],
                                             feed_dict={textRNN.input_x:input_x,
                                                        textRNN.input_y:input_y,
                                                        textRNN.dropout_keep_prob:dropout_keep_prob})
            print('loss:{}  acc:{}  label:{} predict:{}'.format(loss,acc,input_y,predict))


if __name__ == '__main__':
    test()