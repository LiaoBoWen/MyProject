# import tensorflow as tf
# from tensorflow.contrib import rnn
#
# class General_model:
#     def __int__(self,rnn_units,num_layer,encode_embed_size,vocab_size,):
#         self.rnn_units = rnn_units
#         self.num_layer = num_layer
#         self.vocab_size = vocab_size
#         self.encoder_embedding_size = encode_embed_size
#         self.add_placeholder()
#
#
#     def add_placeholder(self):
#         self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
#         self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
#         self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
#         self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')
#
#         # 定义target序列最大长度
#         self.target_seq_len = tf.placeholder(tf.int32, [None, ], name='target_seq_len')
#         self.max_target_seq_len = tf.placeholder(tf.int32, name='maxtarget_seq_len')
#         self.source_seq_len = tf.placeholder(tf.int32, [None, ], name='source_seq_len')
#
#         with tf.variable_scope('embedding'):
#             self.Embedding = tf.get_variable('Embedding',[self.vocab_size,self.encoder_embedding_size],
#                                              initializer=tf.random_normal_initializer(stddev=0.1))
#
#     def encoder(self):
#         encoding_input = tf.nn.embedding_lookup(self.Embedding,self.inputs)
#
#         lstm_fw_cell = rnn.LSTMCell(self.rnn_units)
#         lstm_bw_cell = rnn.LSTMCell(self.rnn_units)
#
#         lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
#         lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
#
#         encode_output, ((fw_state_c,fw_state_h),(bw_state_c,bw_state_h)) = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,
#                                                                       encoding_input,dtype=tf.float32,
#                                                                       ) # todo 加入sequence_length
#         self.encode_output = tf.concat(encode_output,axis=2)
#         encode_state_c = tf.concat([fw_state_c,bw_state_c],axis=1)
#         encode_state_h = tf.concat([fw_state_h,bw_state_h],axis=1)
#         self.encode_state = (encode_state_c,encode_state_h)
#
#     def decoder(self):
#         target_vocab_size = len(target_vocab_size)



import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import LSTMStateTuple
import numpy as np
import json
import os

class Model:
    def __init__(self,vocab_to_int,batch_size,hidden_dim,embeddings,learning_rate,vocab_size):
        self.batch_size = batch_size
        self.vocab_to_int = vocab_to_int
        self.hidden_dim = hidden_dim
        self.embeddings = tf.Variable(embeddings,trainable=True,dtype=tf.float32,name='embeddings')
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size


        self.add_placeholder()
        self.train()




    def add_placeholder(self):
        self.input_data = tf.placeholder(tf.int32,[None,None],name='inputs')
        self.text_length = tf.placeholder(tf.int32,[None,],name='text_length')
        self.targets = tf.placeholder(tf.int32,[None,None],name='targets')
        self.summary_length = tf.placeholder(tf.int32,[None,],name='summary_length')
        self.max_summary_length = tf.reduce_max(self.summary_length,name='max_summary')
        self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')



    def process_encoding_input(self,target_data):
        # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
        ending = tf.strided_slice(target_data,[0,0],[self.batch_size,-1],[1,1])
        dec_input = tf.concat([tf.fill([self.batch_size,1],self.vocab_to_int['<GO>']),ending],axis=1)

        return dec_input

    def encoding_layer(self,sequence_length,rnn_input,keep_prob):
        with tf.variable_scope('encode'):
            cell_fw = rnn.LSTMCell(self.hidden_dim,initializer=tf.random_normal_initializer(-0.1,0.1,seed=2))
            cell_fw = rnn.DropoutWrapper(cell_fw,input_keep_prob=keep_prob)

            cell_bw = rnn.LSTMCell(self.hidden_dim,initializer=tf.random_normal_initializer(-0.1,0.1,seed=2))
            cell_bw = rnn.DropoutWrapper(cell_bw,input_keep_prob=keep_prob)

            enc_out, (enc_state_f,enc_state_b) = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,rnn_input,
                                                                 sequence_length,dtype=tf.float32)

        enc_out = tf.concat(enc_out,axis=2)
        enc_state = (tf.concat([enc_state_f[0],enc_state_b[0]],axis=1),
                     tf.concat([enc_state_f[1],enc_state_b[1]],axis=1))

        return enc_out,enc_state

    def training_decoding_layer(self,dec_embed_input,summary_len,dec_cell,initial_state,output_layer,
                                max_summary_len):
        '''
        :param dec_embed_input:
        :param summary_len:
        :param dec_cell: todo
        :param initial_state:
        :param output_layer:  todo
        :param max_summary_len:
        :return:
        '''
        # inputs is taget but infer<t-1>
        training_helper = seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                 sequence_length=summary_len,
                                                 time_major=False)
        training_decoder = seq2seq.BasicDecoder(dec_cell,training_helper,
                                                initial_state,output_layer)
        training_logits, _, _= seq2seq.dynamic_decode(training_decoder,output_time_major=False,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_len)

        return training_logits

    def inference_decoding_layer(self,embeddings,start_token,end_token,dec_cell,
                                 initial_state,output_layer,max_summary_len):
        start_tokens = tf.tile(tf.constant([start_token],dtype=tf.int32),[self.batch_size],name='start_token')

        inference_helper = seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,end_token)

        inference_decoder = seq2seq.BasicDecoder(dec_cell,
                                                 inference_helper,
                                                 initial_state,
                                                 output_layer)

        inference_logits, _, _ = seq2seq.dynamic_decode(inference_decoder,impute_finished=True,
                                                        maximum_iterations=max_summary_len)

        return inference_logits

    def decoding_layer(self,dec_embed_input,embeddings,enc_output,enc_state,
                       vocab_size,text_len,summary_len,max_sum_len):

        lstm = rnn.LSTMCell(self.hidden_dim * 2,initializer=tf.random_normal_initializer(-0.1,0.1,seed=2))

        dec_cell = rnn.DropoutWrapper(lstm,input_keep_prob=self.keep_prob,)

        output_layer = tf.layers.Dense(vocab_size,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        attn_mech = seq2seq.BahdanauAttention(self.hidden_dim * 2,
                                              enc_output,
                                              text_len,
                                              normalize=False,name='BahdanauAttention')

        dec_cell = seq2seq.AttentionWrapper(dec_cell,attn_mech,attention_layer_size=self.hidden_dim * 2)

        # initial_state = seq2seq.AttentionWrapperState(enc_state[0],_zero_state_tensors(self.hidden_dim,batch_size,
        #                                                                                tf.float32))
        initial_state = dec_cell.zero_state(self.batch_size,tf.float32).clone(cell_state=LSTMStateTuple(*enc_state))

        with tf.variable_scope('decode'):
            traing_logits = self.training_decoding_layer(dec_embed_input,summary_len,dec_cell,initial_state,
                                                         output_layer,max_sum_len)

        with tf.variable_scope('decode',reuse=True):
            inference_logits = self.inference_decoding_layer(embeddings,self.vocab_to_int['<GO>'],
                                                             self.vocab_to_int['<EOS>'],dec_cell,
                                                             initial_state,output_layer,max_sum_len)

        return traing_logits, inference_logits

    def seq2seq_model(self,text_len,target_data,summary_len,max_sum_len,vocab_size):
        enc_embed_input = tf.nn.embedding_lookup(self.embeddings,self.input_data)
        enc_output, enc_state = self.encoding_layer(text_len,enc_embed_input,self.keep_prob)

        dec_input = self.process_encoding_input(target_data=target_data)
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings,dec_input)

        training_logits, inference_logits = self.decoding_layer(dec_embed_input,self.embeddings,
                                                                enc_output,enc_state,vocab_size,
                                                                text_len,summary_len,max_sum_len)

        return training_logits, inference_logits



    def train(self):
        self.global_step = tf.train.get_or_create_global_step()
        training_logit, inference_logit = self.seq2seq_model \
            (self.text_length, self.targets, self.summary_length, self.max_summary_length,self.vocab_size)

        # create tensor for train_logit and inference_logit
        self.training_logits = tf.identity(training_logit.rnn_output, 'logits')
        self.inference_logits = tf.identity(inference_logit.sample_id, 'predictions')

        # create weights for sequence_loss
        masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32, name='masks')

        with tf.variable_scope('optimization'):
            self.cost = seq2seq.sequence_loss(self.training_logits,
                                         self.targets,
                                         masks)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = optimizer.compute_gradients(self.cost)
            cliped_gradients = [(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(cliped_gradients,global_step=self.global_step)




def pad_sentence_batch(sentence_batch):
    max_setence_len = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_setence_len - len(sentence)) for sentence in
            sentence_batch]


def get_batches(texts, summarys, batch_size):
    sum_len = len(summarys)
    batch_num = sum_len // batch_size
    for i in range(batch_num):
        start = i * batch_size
        end = min(i * batch_size + batch_size, sum_len)
        texts_batch = texts[start:end]
        summarys_batch = summarys[start:end]
        pad_summarys_batch = np.array(pad_sentence_batch(summarys_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        summary_len = []
        for summary in summarys_batch:
            summary_len.append(len(summary))

        texts_len = []
        for text in texts_batch:
            texts_len.append(len(text))

        yield pad_texts_batch, pad_summarys_batch, texts_len, summary_len

def invert_to_id(texts):
    text_int = [[vocab_to_int.get(word,vocab_to_int['<UNK>']) for word in text] + [vocab_to_int['<EOS>']] for text in texts]
    return text_int


def read_data(data_path='./data/Time Dataset.json'):
    with open(data_path,'r',encoding='utf8') as f:
        data = json.load(f)
    texts = []
    summarys = []
    for _ in data:
        texts.append(_[0])
        summarys.append(_[1])

    return texts, summarys

epochs = 50
batch_size = 100
hidden_size = 200
learning_rate = 0.005
keep_drop = 0.5
embeddings = np.random.uniform(-0.25,0.25,[20,200])
embeddings = np.float32(embeddings)

from string import ascii_letters
chars = list(ascii_letters)
chars = list(ascii_letters) + [' ','<PAD>','<UNK>','<GO>','<EOS>',':','.',''] + list('1234567890')
int_to_vocab = dict(enumerate(chars))
vocab_to_int = dict(zip(int_to_vocab.values(),int_to_vocab.keys()))


texts, summarys = read_data()

texts = invert_to_id(texts)
summarys = invert_to_id(summarys)


save_path = './model_saved/checkpoint/'

if not os.path.exists('./model_saved'):
    os.makedirs('./model_saved')
if not os.path.exists(save_path):
    os.makedirs(save_path)



def train():
    with tf.Session() as sess:
        model = Model(vocab_to_int,batch_size,hidden_size,embeddings,learning_rate,len(vocab_to_int))
        saver = tf.train.Saver(max_to_keep=3)

        sess.run(tf.global_variables_initializer())

        loss_ = 10000
        for epoch_i in range(epochs):
            for batch_i, (source_batch, target_batch, source_len, target_len) in enumerate(get_batches(texts,summarys,batch_size)):

                _, loss,predicts = sess.run([model.train_op,model.cost,model.inference_logits],feed_dict={model.input_data:source_batch,
                                                                          model.targets:target_batch,
                                                                          model.text_length:source_len,
                                                                          model.summary_length:target_len,
                                                                          model.keep_prob:keep_drop})
                if loss < loss_:
                    loss_ = loss
                    saver.save(sess,save_path + 'check',model.global_step)
            ans = []
            for pred in predicts:
                _ = ''
                for i in pred:
                    _ += int_to_vocab[i]
                ans.append(_)
            ans_ = []
            for pred in target_batch:
                _ = ''
                for i in pred:
                    _ += int_to_vocab[i]
                ans_.append(_)

            print('Epoch:{} loss:{} \npredicts:{} \nreal    :{}\n\n'.format(epoch_i + 1,loss,ans,ans_))




if __name__ == '__main__':
    train()