# import numpy as np
#
# import tensorflow as tf
# from tensorflow import contrib
# from tensorflow.contrib import rnn
# from tensorflow.contrib.rnn import DropoutWrapper
# from utils import *
#
# BATCH_SIZE = config['batch_size']
# unit_num = embedding_size
# time_step = max_sequence
# DROPOUT_RATE = config['dropout']
# EPOCH = config['epoch']
# TAGS_NUM = get_class_size() # todo what?
#
# class NER_net:
#     def __init__(self,scope_name,iterator,embedding,batch_size):
#         """
#         :param scope_name:
#         :param iterator:tesorflow DateSet API把数据偶读feed进来
#         :param embedding: 预训练的word embedding
#         :param batch_size:
#         """
#         self.batch_size = batch_size
#         self.embedding = embedding
#         self.iterator = iterator
#
#         with tf.variable_scope(scope_name) as scope:
#             self._build_net()
#
#     def _build_net(self):
#         self.global_step = tf.Variable(0,trainable=False)
#         source = self.iterator
#         tgt = self.iterator.target_input
#         # 得到当前batch的长度，不足的会被padding填补
#         max_sequence_in_batch = self.iterator.source_sequence_length
#         max_sequence_in_batch = tf.reduce_max(max_sequence_in_batch)
#         max_sequence_in_batch = tf.to_int32(max_sequence_in_batch)
#
#         self.x = tf.nn.embedding_lookup(self.embedding,source)
#         self.y = tgt
#
#         cell_fw = rnn.BasicLSTMCell(num_units=unit_num)
#         cell_bw = rnn.BasicLSTMCell(num_units=unit_num)
#
#         if DROPOUT_RATE is not None:
#             cell_fw = DropoutWrapper(cell_fw,output_keep_prob=DROPOUT_RATE)
#             cell_bw = DropoutWrapper(cell_bw,output_keep_prob=DROPOUT_RATE)
#
#         outputs, states =  tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,self.x,dtype=tf.float32)
#         output = tf.concat(outputs,axis=2)
#
#         W = tf.get_variable('projection_w',[2 * unit_num, TAGS_NUM])
#         b = tf.get_variable('projection_b',[TAGS_NUM])
#
#         x_reshape = tf.reshape(output,[-1,2 * unit_num])
#         projection = tf.matmul(x_reshape,W) + b
#
#         # -1 to time step
#         self.outputs = tf.reshape(projection,[self.batch_size,-1,TAGS_NUM])
#         self.seq_length = tf.convert_to_tensor(self.batch_size * [max_sequence_in_batch],dtype=tf.int32)
#         # todo CRF 登场！
#         self.log_likelihood, self.transition_params = contrib.crf.crf_log_likelihood(
#             self.outputs,self.y,self.seq_length
#         )
#         # add training op to tune the paramters
#         self.loss = tf.reduce_mean(-self.log_likelihood)
#         self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
#
# def train(net,iterator,sess):
#     saver = tf.train.Saver()
#     ckpt =  tf.train.get_checkpoint_state(model_path)   # todo ???
#     if ckpt is not None:
#         path = ckpt.model_checkpoint_path
#         print('loading pre_trained model form {}'.format(path))
#         saver.restore(sess,path)
#
#     current_epoch = sess.run(net.global_step)
#     while True:
#         if current_epoch > EPOCH : break
#         try:
#             tf_unary_scores, tf_transition_params, _, losses = sess.run(
#                 [net.outputs,net.transition_params.net.train_op,net.loss])
#             if current_epoch % 100 == 0:
#                 print('*' * 100)
#                 print(current_epoch,'loss',losses)
#                 print('*' * 100)
#             if current_epoch % (EPOCH / 10) == 0 and current_epoch:
#                 sess.run(tf.assign(net.global_step,current_epoch))
#                 saver.save(sess,model_path + 'points', global_step=current_epoch)
#
#             current_epoch += 1
#         except tf.errors.OutOfRangeError:
#             sess.run(iterator.initializer)
#         except tf.errors.InvalidArgumentError:
#             sess.run(iterator.initializer)
#     print('training finished ')
#
# def predict(net,tag_table,sess):
#     saver = tf.train.Saver()
#     ckpt = tf.train.get_checkpoint_state(model_path)
#     if ckpt is not None:
#         path = ckpt.model_checkpoint_path
#         print('loading pre-trained model from {}...'.format(path))
#         saver.restore(sess,path)
#     else:
#         print('Model not found, please train your model first')
#         return
#
#     # 获取原文的iterator
#     file_path = file_content_iterator(pred_file)
#
#     while True:
#         # batch等于1的时候本就没有padding，如果皮脸预测的话要做长度的截取
#         try:
#             tf_unary_scores, tf_transition_params = sess.run([net.outputs,net.transition_params])
#         except tf.errors.OutOfRangeError: # todo 这是什么的错
#             print('Prediction finished')
#             break
#
#         # 把batch所在维度去掉
#         tf_unary_scores = np.squeeze(tf_unary_scores)
#
#         # viter算法登场
#         viterbi_sequence, _ = contrib.crf.viterbi_decode(
#             tf_unary_scores, tf_transition_params)
#         tags = []
#         for id in viterbi_sequence:
#             tags.append(sess.run(tag_table.lookup(tf.constant(id,dtype=tf.int64))))
#         write_result_to_file(file_path,tags)
#
# if __name__ == '__main__':
#     action = config['action']
#     # 获取词的数目
#     vocab_size = get_src_vocab_size()
#     src_unknow = tgt_unknown_id = vocab_size
#     src_padding = vocab_size + 1
#
#     src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_voacb_file, src_unknown_id, tgt_unknown_id)
#     embedding = load_word2vec_embedding(vocab_size)
#
#     if action == 'train':
#         iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
#     elif action == 'predict':
#         BATCH_SIZE = 1
#         DRoPOUT_RATE = 1.
#         iterator = get_predict_iteratoe(src_vocab_table, vocab_size,BATCH_SIZE)
#     else:
#         print('Only support train and predict actioins.')
#         exit(0)
#
#     tag_table = tag_to_id_table()
#     net = NER_net('ner',iterator,embedding,BATCH_SIZE)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         sess.run(iterator.initializer)
#         tf.tables_initializer().run()   # todo ???
#
#         if action == 'train':
#             train(net,iterator,sess)
#         elif action == 'predict':
#             predict(net,tag_tabls,sess)


import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Model:
    def __init__(self,config,embedding_pretrained,dropout_keep=1.):
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.embedding_size = config['embedding_size']
        self.embedding_dim = config['embedding_dim']
        self.sen_len = config['sen_len']
        self.tag_size = config['tag_size']
        self.pretrained = config['pretrained']
        self.dropout_keep = dropout_keep
        self.embedding_pretrained = embedding_pretrained
        self.input_data = tf.placeholder(tf.int32,shape=[self.batch_size,self.sen_len],name='input_data')
        self.labels = tf.placeholder(tf.int32,shape=[self.batch_size,self.sen_len],name='labels')
        self.embedding_placeholder = tf.placeholder(tf.float32,shape=[self.embedding_size,self.embedding_dim],name='embedding_placeholder')
        with tf.variable_scope('bilstm_crf') as scope:
            self._build_net()

    def _build_net(self):
        word_embeddings = tf.get_variable('word_embeddings',[self.embedding_size,self.embedding_dim])
        if self.pretrained:
            embedding_init = word_embeddings.assign(self.embedding_pretrained)
        input_embedded = tf.nn.embedding_lookup(word_embeddings,self.input_data)
        input_embedded = tf.nn.dropout(input_embedded,self.dropout_keep)   # todo 这么快开始dropout?

        lstm_fw = rnn.BasicLSTMCell(self.embedding_dim)
        lstm_bw = rnn.BasicLSTMCell(self.embedding_dim)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,input_embedded,dtype=tf.float32,time_major=False,scope=None)
        bilstm_output = tf.concat(outputs,axis=2)

        # Fc
        W = tf.Variable(tf.truncated_normal([self.batch_size, 2 * self.embedding_dim, self.tag_size],dtype=tf.float32),name='W')
        b = tf.Variable(tf.zeros([self.batch_size,self.sen_len,self.tag_size],dtype=tf.float32),name='b')
        bilstm_out = tf.tanh(tf.matmul(bilstm_output,W) + b)

        # Linear-CRF
        from tensorflow.contrib.crf import crf_log_likelihood
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_output,self.labels,tf.tile(np.array([self.sen_len]),np.array(self.batch_size)))

        loss = tf.reduce_mean(-log_likelihood)

        # calculate the viterbi sequence and score (used for prediction and test time)
        self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out,self.transition_params,tf.tile(np.array[self.sen_len]),np.array(self.batch_size))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)