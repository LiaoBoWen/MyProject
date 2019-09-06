# import tensorflow as tf
# from tensorflow.python.ops import lookup_ops # todo ????
# import numpy as np
# import collections
# from config import config
# import os
#
# src_file = config['src_file']
# tgt_file = config['tgt_file']
# # 预测时使用
# predict_file = config['predict_file']
# src_vocab_file = config['src_vocab_file']
# tgt_vocab_file = config['tgt_vocab_file']
# word_embedding_file = config['word_embedding_file']
# model_path = config['model_path']
# embedding_size = config['embedding_size']
# max_sequence = config['max_sequence']
#
# class BatchedInput(collections.namedtuple('BatchedInput',
#                                           ('initializer',
#                                            'source',
#                                            'target_input',
#                                            'source_sequence_length',
#                                            'target_sequence_length'))):
#     pass
#
# def build_word_index():
#     '''生成单词列表，并存入文件之中'''
#     if not os.path.exists(word_embedding_file):
#         print('word embedding file does not exits,please check your path')
#         return
#
#     print('building word index...')
#     if not os.path.exists(src_file):
#         with open(src_vocab_file,'w') as source:
#             f = open(word_embedding_file,'r',encoding='utf8')
#             for line in f:
#                 values = line.split()
#                 word = values[0] # 取词
#                 source.write(word + '\n')
#         f.close()
#     else:
#         print('source vocabulary file has already existed,continue to next stage.')
#
#     if not os.path.exists(tgt_vocab_file):
#         with open(tgt_file,'r') as source:
#             dict_word = {}
#             for line in source.readlines():
#                 line = line.strip()
#                 if line != '':
#                     word_arr = line.split()
#                     for w in word_arr:
#                         dict_word[w] = dict_word.get(w,0) + 1
#             top_words = sorted(dict_word.items(),key=lambda x: x[1] ,reverse=True)
#             with open(tgt_vocab_file,'w') as s_vocab:
#                 for word, frequence in top_words:
#                     s_vocab.write(word + '\n')
#     else:
#         print('target vocabulary file has alreadly existed, continue to next stage')
#
#     if not os.path.exists(model_path):
#         os.mkdir(model_path)
#
# def get_src_vocab_size():
#     '''
#     :return: 训练数据集中有多少不重复的词
#     '''
#     size = 0
#     with open(src_vocab_file) as vocab_file:
#         for content in vocab_file.readlines():
#             content = content.strip()
#             if content != '':
#                 size += 1
#     return size
#
# def get_class_size():
#     '''获取命名实体识别类别总数'''
#     size = 0
#     with open(tgt_vocab_file) as vocab_file:
#         for content in vocab_file.readlines():
#             if content.strip() != '':
#                 size += 1
#     return size
#
# def create_vocab_tables(src_vocab_file,tgt_vocab_file,src_unknown_id,tgt_unknown_id,share_vocab=False):
#     src_vocab_table = lookup_ops.index_table_from_file(
#         src_vocab_file, default_value=src_unknown_id)
#     if share_vocab:
#         tgt_vocab_file = lookup_ops.index_table_from_file(tgt_vocab_file,default_value=tgt_unknown_id)
#     return src_vocab_table,tgt_vocab_file
#
# def get_iterator(src_vocab_table, tgt_vocab_table, vocab_size,batch_size,buffer_size=None,random_seed=None,
#                  num_threads=8,src_max_len=max_sequence,num_bucket=5):
#     if buffer_size is None:
#         # 如果buffer_size比总数大很多，会报End of sequence warning.
#         #  # https://github.com/tensorflow/tensorflow/issues/12414
#         buffer_size = batch_size * 30
#
#
#     src_dataset = tf.data.TextLineDataset(src_file)
#     tgt_dataset = tf.data.TextLineDataset(tgt_file)
#     src_tgt_dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))
#
#     src_tgt_dataset = src_tgt_dataset.shuffle(buffer_size,random_seed)
#     src_tgt_dataset = src_tgt_dataset.map(lambda src,tgt :(tf.string_split([src]).values,tf.string_split([tgt]).values),
#                                           num_parallel_calls=num_threads)
#     src_tgt_dataset.prefetch(buffer_size)
#
#     if src_max_len:
#         src_tgt_dataset = src_tgt_dataset.map(
#             lambda src,tgt:(src[])
#         )




import re
import numpy as np

def calculate(x,y,id2word,id2tag,res=[]):
    entity = []
    for i in range(len(x)):
        for j in range(x[0]):
            if x[i][j] == 0 or y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = [id2word[x[i][j]] + '/' + id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity = []
            else:
                entity = []
    return res

def get_entity(x,y,id2tag):
    entity = ''
    res = []
    for i in range(len(x)): # sentences
        for j in range(len(x[0])): # words
            if y[i][j] == 0:
                continue
            if id2tag[y[i][j]] == 'B':
                entity = id2tag[y[i][j]][1:] + ':' + x[i][j]
            elif id2tag[y[i][j]][0] == "M" and len(entity) != 0:
                entity += x[i][j]
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0:
                entity += x[i][j]
                res.append(entity)
                entity = []
            else:
                entity = []
    return res

def write_entity(output,x,y,id2tag):
    '''每次使用在文档的最后添加新信息'''
    entity =''
    for i in range(len(x)):
        for j in range(len(x[0])):
            if y[i] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = id2tag[y[i]][2:] + ':' + x[i]
            elif id2tag[y[i][0]] == "M" and len(entity) != 0:
                entity += x[i]
            elif id2tag[y[i][0]] == 'E' and len(entity) != 0:
                entity += x[i]
                output.write(entity + ' ')
                entity = ''
            else:
                entity = ''
    return

def train(model,sess,saver,epoches,batch_size,data_train,data_test,id2word,id2tag):
    batch_num = int(data_train.y.shape[0] / batch_size)
    batch_num_test = int(data_test.y.test.shape[0] / batch_size)
    for epoch in range(epoches):
        for batch in range(batch_num):
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict =  {model.input_data:x_batch,model.labels:y_batch}
            pre, _ = sess.run([model.viterbi_sequence,model.train_op],feed_dict)
            acc = 0
            if batch % 200 == 0:
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j] == pre[i][j]:
                            acc += 1
                print(acc / (len(y_batch) * len(y_batch[0])))
        path_name = './model/model{}.ckpt'.format(epoch)
        print(path_name)
        if epoch % 3 == 0:
            saver.save(sess,path_name)
            print('model hsa been saved')
        entityres = []
        entityall = []
        for batch in range(batch_num):
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {model.input_data:x_batch,model.labels:y_batch}
            pre = sess.run([model.viterbi_sequence],feed_dict)
            pre = pre[0]
            entityres = calculate(x_batch,pre,id2word,id2tag,entityres)
            entityall = calculate(x_batch,y_batch,id2word,id2tag,entityall)
        jiaoji = [i for i in entityres if i in entityall]
        # jiaoji = list(set(entityres) & set(entityall))
        if len(jiaoji) != 0:
            zhun = len(jiaoji) / len(entityres)
            zhuo = len(jiaoji) / len(entityall)
            print('train')
            print('zhun: {}'.format(zhun))
            print('zhuo: {}'.format(zhuo))
        else:
            print('zhun: 0')
        entityres = []
        entityall = []
        for batch in range(batch_num_test):
            x_batch, y_batch = data_test.next_batch(batch_size)
            feed_dict ={model.input_data:x_batch,model.labels:x_batch}
            pre = sess.run([model.viterbi_sequence],feed_dict)
            pre = pre[0]
            entityres = calculate(x_batch,pre,id2word,id2tag,entityres)
            entityall = calculate(x_batch,y_batch,id2word,id2tag,entityall)
        jiaoji = [i for i in entityres if i in entityall]
        if len(jiaoji) != 0:
            zhun = len(jiaoji) / len(entityres)
            zhuo = len(jiaoji) / len(entityall)
            print('test')
            print('zhun: {}'.format(zhun))
            print('zhuo: {}'.format(zhuo))
        else:
            print('zhun: 0')


max_len = 60
def padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        ids.extend([0] * (max_len - len(ids)))
        return ids

def padding_word(sen):
    if len(sen) >= max_len:
        return sen[:max_len]
    else:
        return sen

def test_input(model,sess,word2id,id2tag,batch_size):
    while True:
        text = input('Enter your input :')
        text = re.split(r'[，。！？、”“‘’（）]',text)
        text_id = []
        for sen in text:
            word_id = []
            for word in sen:
                if word in word2id:
                    word_id.append(word2id['unknow'])
            text_id.append(padding(word_id))
        zero_padding = []
        zero_padding.extend([0] * max_len)
        text_id.extend([zero_padding] * (batch_size - len(text_id)))
        feed_dict = {model.input_data:text_id}
        pre = sess.run(model.viterbi_sequence,feed_dict)
        entity = get_entity(text,pre[0],id2tag)
        print('result')
        for i in entity:
            print(i)

def extraction(input_path,output_path,model,sess,word2id,id2word,id2tag,batch_size):
    text_id = []
    text = []
    with open(input_path,'rb',encoding='utf8') as inp:
        for line in inp.readlines():
            line = re.split(r'[，。？！、”“‘’（）]',line.strip())
            for sentence in line:
                if sentence == '' or sentence == ' ':
                    continue
                word_id = []
                for word in sentence:
                    if word in word2id:
                        word_id.append(word2id['unknow'])
                    else:
                        word_id.append(padding(word_id))
                text_id.append(padding(word_id))
                text.append(padding_word(sentence))
    zero_padding = []
    zero_padding.extend([0] * max_len)
    text_id.extend([zero_padding] * (batch_size -len(text_id)) % batch_size)
    text_id = np.asarray(text_id)
    text_id = text_id.reshape(-1,batch_size,max_len)
    predict = []
    for index in range(len(text_id)):
        feed_dict = {model.input_data:text_id[index]}
        pre = sess.run([model.viterbi_sequence],feed_dict)
        predict.append(pre[0])
    predict = np.asarray(predict).reshape(-1,max_len)

    with open(output_path,'a',encoding='utf8') as oup:
        for index in range(len(text)):
            oup.write(text[index] + '  ')
            write_entity(oup,text[index],predict[index],id2tag)
            oup.write('\n')