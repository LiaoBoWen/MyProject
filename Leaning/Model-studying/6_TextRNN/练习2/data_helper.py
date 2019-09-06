import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')

import os
import pickle
import logging
import numpy as np
from collections import Counter
from gensim.models import word2vec,KeyedVectors
from tflearn.data_utils import pad_sequences

logging.basicConfig(level=logging.INFO)

PAD_ID = 0
_GO = '_GO'
_END = '_END'
_PAD = '_PAD'

def load_data_multilabel_new():
    pass

def create_vocab(word2vec_model_path='../data/zhihu-word2vec-title-desc.bin-100.txt',scope=''):
    cache_path = './cache_vocab_label_pkl/' + scope + '_word_vocab.pkl'
    logging.info('cache_path :{}\tfile_exists:{}'.format(cache_path, os.path.exists(cache_path)))

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            word2idx, idx2word = pickle.load(f)
            return word2idx, idx2word
    else:
        word2idx = {}
        idx2word = {}
        model = KeyedVectors.load_word2vec_format(word2vec_model_path)
        word2idx['PAD'] = 0
        idx2word[0] = 'PAD'

        if 'biLstmTextRelation' in scope:
            word2idx['EOS'] = 1
            idx2word[1] = 'EOS'
        for i, value in enumerate(model.vocab):
            word2idx[value] = i
            idx2word[i] = value
        if not os.path.exists(cache_path):
            with open(cache_path, 'wb') as f:
                pickle.dump((word2idx, idx2word), f)
    return word2idx, idx2word


def create_vocab_label(vocab_label='../data/train-zhihu4-only-title-all.txt',scope='',use_se2seq=False):
    print('create_vocab.train_path:{}'.format(vocab_label))
    cache_path = './cache_vocab_label_pkl/' + scope + '_label.pkl'

    if os.path.exists(cache_path):
        with open(cache_path,'rb') as f:
            word2idx_label, idx2word_label = pickle.load(f)
        return word2idx_label,idx2word_label

    else:
        results = Counter()
        with open(vocab_label,'r',encoding='utf8') as f:
