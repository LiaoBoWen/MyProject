import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')

import codes
import numpy as np

from gensim.models import word2vec, KeyedVectors

import os
import pickle
PAD_ID = 0
from tflearn.data_utils import pad_sequences
from keras.preprocessing.sequence import pad_sequences

_GO = '_GO'
_END = '_END'
_PAD = '_PAD'

def create_vocabulary(simple=None,word2vec_model_path='./data/zhihu-word2vec-title-desc.bin-100.txt',name_scope=''):
    cache_path = 'cache_vocabulary_label_pik/' + name_scope
    print('cache_path:',cache_path,'file_exists:',os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path,'rb') as data_f:
            word2index, index2word = pickle.load(data_f)
            return word2index, index2word
    else:
        word2index = {}
        index2word = {}

        print('create vocabulary. word2vec_model_path:',word2vec_model_path)
        model = KeyedVectors.load_word2vec_format(word2vec_model_path,binary=True)
        word2index['PAD_ID'] = 0
        index2word[0] = 'PAD_ID'
        special_index = 0
        if 'biLstmTextRelation' in name_scope:
            word2index['EOS'] = 1
            index2word[1] = 'EOS'
            special_index = 1
        for i, vocab in enumerate(model.vocab):
            word2index[vocab] = i
            index2word[i] = vocab

        if not os.path.exists(cache_path):
            with open(cache_path,'ab') as data_f:
                pickle.dump((word2index,index2word),data_f)
    return word2index, index2word

def create_vocabulary_label(vocabulary_label='./data/train-zhihu4-only-title-all.txt',name_scope='',use_seq2seq=False):
    print('create_vocabulary_labrl_storted.started.training_data_path:',vocablary_label)
    cache_path = 'cache_vocabulary_label_pik/' + name_scope + '_label_vocabulary.pik'
    if os.path.exists(cache_path):
        with open(cache_path,'rb') as data_f:
            word2index_label, index2word_label = pickle.load(data_f)
            return word2index_label, index2word_label
    else:
        zhihu_f_train = codes.open(vocabulary_label,'r','utf8')
        lines = zhihu_f_train.readlines()
        count = 0
        word2index_label = {}
        index2word_label = {}
        label_count_dict = {}
        for i,line in enumerate(lines):
            if '__label__' in line:
                label = line[line.index('__label__') + len("__label__"):].strip().replace('\n','')
                if label_count_dict.get(label,None) is not None:
                    label_count_dict[label] = label_count_dict[label] + 1
                else:
                    label_count_dict[label] = 1 # UNK = 1
        list_label = sort_by_value(label_count_dict)

        print('length of list_label:',len(list_label))

        count = 0

        if use_seq2seq:
            i_list = [0,1,2]
            label_special_list = [_GO,_END,_PAD]
            for i, label in zip(i_list,label_special_list):
                word2index_label[label] = i
                index2word_label[i] = label

        for i, label in enumerate(list_label):
            if i < 10:
                count_value = label_count_dict[label]
                print('label:',label,'count_value:',count_value)
                count = count + count_value
            index = i + 3 if use_seq2seq else i
            word2index_label[label] = index
            index2word_label[index] = label
        print('count top10:',count)

        if not os.path.exists(cache_path):
            with open(cache_path,'ab') as data_f:
                pickle.dump((word2index_label,index2word_label),data_f)
    print('create_vocabulary_label_sorted.ended.len of vocvabulary_label:',len(index2word_label))
    return word2index_label,index2word_label

def sort_by_value(d):
    items = d.items()
    backitems = [v[1]. v[0] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(len(backitems))]

def load_data_multiable_new(word2index,word2index_label,valid_portion=0.05,max_training_data=1e6,
                            training_data_path='train-zhihu4-only-title-all.txt',multi_label_flag=True,use_seq2seq=False,seq2seq_label_length=6):
    # 加载 zhihu data from file
    print('load_data.started...')
    print('load_data_multi_newtraining_data_path:',training_data_path)
    zhihu_f = codes.open(training_data_path,'r','urf8')
    lines = zhihu_f.readlines()
    X = []
    Y = []
    Y_decoder_input = []
    for i, line in enumerate(lines):
        x, y = line.split('__label__')
        y = y.strip().replace('\n','')
        x = x.strip()
        if i < 1:
            print(i,'x0:',x)
        if use_seq2seq:
            ys = y.replace('\n','').split(" ")
            _PAD_INDEX = word2index_label[_PAD]
            ys_multihot_list = [_PAD_INDEX] * seq2seq_label_length
            ys_decoder_input =  [_PAD_INDEX] * seq2seq_label_length
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1 :
                    ys_multihot_list[j] = word2index_label
            if len(ys) > seq2seq_label_length - 1:
                ys_multihot_list[seq2seq_label_length - 1] = word2index_label[_END]
            else:
                ys_multihot_list[len(ys)] = word2index_label[_END]

            ys_decoder_input[0] = word2index_label[_GO]
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_multihot_list[j] = word2index_label[y]
            if i < 10 :
                print(i,'ys:======>0',ys)
                print(i,'ys_multihot_list:=======>1',ys_multihot_list)
                print(i,'ys_decoder_input:=======>2',ys_decoder_input)
        else:
            if multi_label_flag:
                ys = y.replace('\n','').split(' ')
                ys_index = []
                for y in ys:
                    y_index = word2index[y]
                    ys_index.append(y_index)
                ys_multihot_list = transform_multiable_as_multihot(ys_index)
            else:
                ys_multihot_list = word2index_label[y]
        if i <= 3:

            print('ys_index:')
            print(i,'y:',y,' ;ys_multihot_list:',ys_multihot_list)
            X.append(x)
            Y.append(ys_multihot_list)
            if use_seq2seq:
                Y_decoder_input.append(ys_decoder_input)
        number_examples = len(X)
        print('number_examples:',number_examples)
        train = (X[:int((1-valid_portion) * number_examples)],Y[:int((1 - valid_portion) * number_examples)])
