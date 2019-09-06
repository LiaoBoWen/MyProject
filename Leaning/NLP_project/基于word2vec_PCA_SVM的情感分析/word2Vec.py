import warnings
warnings.filterwarnings(action='ignore',module='gensim',category=UserWarning)
warnings.filterwarnings(action='ignore',module='gensim',category=FutureWarning)

import os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import pickle
import jieba

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
from tensorflow.contrib import learn

def data_helper(neg='./data/neg_split.txt',pos='./data/pos_split.txt'):
    with open(neg,encoding='utf8') as neg_:
        with open(pos,encoding='utf8') as pos_:
            neg_content = neg_.readlines()
            neg_content = [x.strip() for x in neg_content][:-1]
            pos_content = pos_.readlines()
            pos_content = [x.strip() for x in pos_content][:-1]

            neg_len = len(neg_content)
            pos_len = len(pos_content)

            x = pos_content + neg_content

            y = np.concatenate([np.ones(pos_len), np.zeros(neg_len)]).reshape([-1,1])

    return x, y


def makeVec(x,max_len=400,vocab_path='./data/vocab.pkl',word2Vec='./data/w2v.pkl'):
    vocab = learn.preprocessing.VocabularyProcessor(max_len)
    if not os.path.exists(vocab_path):
        x_id = vocab.fit_transform(x)
        vocab.save(vocab_path)
    else:
        vocab = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_id = vocab.transform(x)
    x_id = np.array(list(x_id)).astype(str)
    x_id = [list(_) for _ in x_id]

    if not os.path.exists(word2Vec):
        w2v = Word2Vec(size=8,window=5,min_count=0,workers=cpu_count())
        w2v.build_vocab(x_id)
        print(w2v.corpus_count)
        w2v.train(x_id,total_examples=w2v.corpus_count,epochs=100)
        w2v.save(word2Vec)
    else:
        w2v = Word2Vec.load(word2Vec)

    x = [np.mean([w2v[w] for w in s],axis=0) for s in x_id]

    return x

def lineWord(x):
    with open('./data/all.txt',encoding='utf8') as f:
        with open('./data/all_split.txt','a',encoding='utf8') as fs:
            text = f.read()
            sentences_ = text.split('\n')
            for sentence in sentences_:
                fs.write(' '.join(jieba.cut(sentence)) + '\n')

    sentences = word2vec.LineSentence('./data/all_split.txt')
    model = Word2Vec(sentences=sentences,window=3,min_count=1,workers=cpu_count(),size=8)
    print(model.wv.vocab)
    X = [np.mean([model[w] for w in s.split()],axis=0) for s in x]
    return X



if __name__ == '__main__':
    X, Y = data_helper()
    max_len = max([len(x.split()) for x in X])
    X = makeVec(X,max_len=max_len)
    # X = lineWord(X)

    print('MAKE Word2Vec finished...')
    print(X[:2])
    print(Y[:2])

    Data = np.concatenate([X,Y],axis=1)
    shuffle_indice = np.random.permutation(len(Y))
    Data = Data[shuffle_indice]
    Data = pd.DataFrame(Data)
    print('CONCAT finished...')

    Data.to_csv('Data.csv',header=None)
    print('Saved finished...')