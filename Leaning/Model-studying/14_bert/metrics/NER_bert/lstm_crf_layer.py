import tensorflow as tf
from tensorflow.contrib import crf
from tensorflow.contrib import rnn

class Bilstm_crf:
    '''bert 训练得到的各个token'''
    def __init__(self, embedded_chars, hidden_units, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training):
        self.hidden_units = hidden_units
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.initializers = initializers
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.labels = labels
        self.lengths = lengths
        self.is_training = is_training
        self.embedded_chars = embedded_chars
        self.embedding_dims = embedded_chars.shape[-1].value


    def add_bilstm_crf_layer(self, crf_only):
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars,keep_prob=self.dropout_rate)

        # 直接进行dense最后tanh激活输出logits
        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        # 通过多层双向的lstm最后输出logits
        else:
            lstm_output = self.bilstm_layer(self.embedded_chars)
            logits = self.project_bilstm_layer(lstm_output)

        loss, trans = self.crf_layer(logits)
        # 非viterbi算法解码
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length= self.lengths)
        return loss, logits, trans, pred_ids


    def bilstm_layer(self,embedded_chars):
        with tf.variable_scope('rnn_layer'):
            f_cell, b_cell = rnn.BasicLSTMCell(self.hidden_units), rnn.BasicLSTMCell(self.hidden_units)
            if self.num_layers > 1:
                f_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.hidden_units) for _ in range(self.num_layers)])
                b_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.hidden_units) for _ in range(self.num_layers)])

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, embedded_chars, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)

        return outputs


    def project_bilstm_layer(self, lstm_outputs, name=None):
        # 分两次进行dense， 第一次hidden_size * 2 -> hidden_size, 第二次hidden_size -> num_labels
        with tf.variable_scope('project' if not name else name):
            with tf.variable_scope('hidden'):
                W = tf.get_variable('W', shape=[self.hidden_units * 2, self.hidden_units], dtype=tf.float32,initializer=self.initializers.xavier_initializer())
                b = tf.get_variable('b', shape=[self.hidden_units], dtype=tf.float32,  initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_units * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            with tf.variable_scope('logits'):
                W = tf.get_variable('W', shape=[self.hidden_units, self.num_labels], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable('b', shape=[self.num_labels], dtype=tf.float32, initializer=tf.zeros_initializer())
                logits = tf.nn.xw_plus_b(hidden, W, b)
        return tf.reshape(logits, [-1, self.seq_length, self.num_labels])


    def project_crf_layer(self, name=None):
        with tf.variable_scope('project' if not name else name):
            with tf.variable_scope('logits'):
                W = tf.get_variable('W', shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable('b', shape=[self.num_labels], dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                output = tf.reshape(self.embedded_chars, shape=[-1, self.embedding_dims])
                logits = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(logits, [-1, self.seq_length, self.num_labels])


    def crf_layer(self, logits):
        with tf.variable_scope('crf_loss'):
            trans = tf.get_variable(
                'transitions',
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likehood, trans = crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths
            )
        return tf.reduce_mean(-log_likehood), trans