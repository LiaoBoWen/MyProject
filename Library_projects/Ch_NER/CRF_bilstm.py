import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


class CRF_bilstm:
    def __init__(self,vocab_size,num_tag,learning_rate,hidden_unit,clip_grad):
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.hidden_unit = hidden_unit
        self.clip_grad = clip_grad
        self.num_tag = num_tag
        self.global_step = tf.train.get_or_create_global_step()

        self.add_placeholder()
        self.bilstm_layer()
        self.loss_layer()
        self.train_layer()



    def add_placeholder(self):
        self.inputs = tf.placeholder(tf.int32,[None,None],name='input_data')
        self.labels = tf.placeholder(tf.int32,[None,None],name='labels')
        self.seq_len = tf.placeholder(tf.int32,[None,],name='seq_len')
        self.dropout_prob = tf.placeholder(tf.float32,name='dropout_prob')

        with tf.variable_scope('embedding'):
            self.Embedding = tf.get_variable('Embedding',shape=[self.vocab_size,self.hidden_unit],
                                             initializer=tf.random_uniform_initializer(-0.25,0.25))

            self.word_embedding_ = tf.nn.embedding_lookup(self.Embedding,self.inputs,name='word_embedding')

            self.word_embedding = tf.nn.dropout(self.word_embedding_,self.dropout_prob)  # todo try removing dropout

        with tf.variable_scope('Projection'):
            self.projection_W = tf.get_variable('projection_W',
                                                [2 * self.hidden_unit,self.num_tag],
                                                initializer=layers.xavier_initializer(),
                                                dtype=tf.float32)

            self.projection_b = tf.get_variable('projection_b',
                                                [self.num_tag],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32)


    def bilstm_layer(self):
        with tf.variable_scope('bilstm-layer'):
            fw_cell = rnn.LSTMCell(self.hidden_unit)
            bw_cell = rnn.LSTMCell(self.hidden_unit)

            # fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=self.dropout_prob) # todo add these
            # bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=self.dropout_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,
                                                              self.word_embedding,
                                                              sequence_length=self.seq_len,
                                                              dtype=tf.float32)
            outputs = tf.concat(outputs,axis=2)
            outputs = tf.nn.dropout(outputs,self.dropout_prob)

            raw_shape = tf.shape(outputs)
            outputs = tf.reshape(outputs,[-1,self.hidden_unit * 2])
            pred = tf.nn.xw_plus_b(outputs,self.projection_W,self.projection_b)

            self.logits = tf.reshape(pred,[-1,raw_shape[1],self.num_tag])


    def loss_layer(self):
        log_likelihood, self.transition_params = crf_log_likelihood(self.logits,
                                                                    self.labels,
                                                                    self.seq_len)

        self.loss = -tf.reduce_mean(log_likelihood)

        tf.summary.scalar('loss',self.loss)


    def train_layer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss)

        grads_and_vars = [(tf.clip_by_value(grad,-self.clip_grad,self.clip_grad),var) for grad, var in grads_and_vars]

        self.train_op = optimizer.apply_gradients(grads_and_vars,global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def dev_decode(self, sess,dev_data, dev_len, labels=None):
        feed_dict = {
            self.inputs: dev_data,
            self.seq_len: dev_len,
            self.dropout_prob: 1.0,
        }
        if labels is not None:
            feed_dict[self.labels] = labels

            loss, logits, transition_params = sess.run([self.loss, self.logits, self.transition_params],
                                                   feed_dict=feed_dict)
            pred_labels = []
            for logit, len_ in zip(logits, dev_len):
                viterbi_seq, _ = viterbi_decode(logit[:len_], transition_params)
                pred_labels.append(viterbi_seq)

            return pred_labels, loss
        else:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)

            pred_labels = []
            for logit, len_ in zip(logits, dev_len):
                viterbi_seq, _ = viterbi_decode(logit[:len_], transition_params)
                pred_labels.append(viterbi_seq)

            return pred_labels

    def precision_and_recall_and_fb1(self,dev_labels):
        '''计算测试集上的效果：计算precision，recall，fb1'''
        dev_label = []

        for _ in dev_labels:
            dev_label.extend(_)

        dev_ORG_lst, dev_LOC_lst, dev_PER_lst = [], [], []
        temp = 0
        temp_lst = []
        for seq in dev_labels:
            temp_org, temp_loc, temp_per = [], [], []
            for i in range(len(seq)):
                if (temp == 0) and ((seq[i] == 1) or (seq[i] == 3) or (seq[i] == 5)):
                    temp = dev_label[i]
                    temp_lst.append(i)
                elif i == len(seq) - 1 or seq[i + 1] == 0 or seq[i + 1] == 1 or seq[i + 1] == 3 or seq[i + 1] == 5:
                    temp_lst.append(i)
                    if temp == 1:
                        temp_org.append(temp_lst)
                    if temp == 3:
                        temp_loc.append(temp_lst)
                    if temp == 5:
                        temp_per.append(temp_lst)
                    temp_lst = []
                    temp = 0
                else:
                    temp_lst.append(i)
            dev_ORG_lst.append(temp_org)
            dev_LOC_lst.append(temp_loc)
            dev_PER_lst.append(temp_per)

        return dev_ORG_lst, dev_LOC_lst, dev_PER_lst
