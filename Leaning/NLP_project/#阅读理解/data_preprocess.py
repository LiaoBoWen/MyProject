import nltk
import jieba
import os
import pickle

from sklearn.metrics import roc_auc_score

from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords,wordnet   # wordnet 构建了单词与单词之间的关系
from nltk.stem import WordNetLemmatizer     # 词性还原工具
from collections import Counter

lemma = WordNetLemmatizer()

raw_data_path = '../data/WikiQA/raw'
save_process_path = './data/WikiQA/processed'

if os.path.exists(save_process_path):
    os.mkdir(save_process_path)


def segement(filename,use_lemma=True):
    '''分词、提取词干'''
    processes_QA = []
    count = 0
    with open(os.path.join(raw_data_path,filename),'r') as f:
        f.readline()
        for line in f.readlines():
            items = line.strip().split('\t')
            qId, Q, aId, A, label = items[0], items[1], items[4], items[5], items[6]
            if use_lemma:
                Q = ' '.join([lemma.lemmatize(_) for _ in jieba.cut(Q)]).lower()
                A = ' '.join([lemma.lemmatize(_) for _ in jieba.cut(A)]).lower()
            else:
                Q = ' '.join(jieba.cut(Q))
                A = ' '.join(jieba.cut(A))
            processes_QA.append([qId,Q,aId,A,label])
            count += 1
            if count % 100:
                print('# Finished {}'.format(count))
    return processes_QA

def build_vocab(corpus,topK=None):
    '''建立词表'''
    counter = Counter()
    for line in corpus:
        counter.update(line[1].split()) # Q->word
        counter.update(line[3].split()) # A->word
    if topK:
        counter = [counter_[0] for counter_ in counter.most_common(topK)]
    else:
        counter = [counter_[0] for counter_ in counter.most_common()]

    counter = {word : idx + 2 for idx, word in counter}
    counter['<PAD>'] = 0
    counter['<UNK>'] = 0
    reverse_counter = dict(zip(counter.values(),counter.keys()))

    return counter, reverse_counter

def transform(corpus,word2idx,unk_id=1):
    '''句子ID化'''
    transformed_corpus = []
    for line in corpus:
        qId, Q, aId, A, label = line
        Q = [word2idx.get(w,unk_id) for w in Q.split()]
        A = [word2idx.get(w,unk_id) for w in A.split()]
        transformed_corpus.append([qId, Q, aId, A, label])

    return transformed_corpus

def pointwise_data(corpus,keep_ids=False):
    pointwise_corpus = []
    for sample in corpus:
        qid, q, aid, a, label = sample
        if keep_ids:
            pointwise_corpus.append([qid, q, aid, a, label])
        else:
            pointwise_corpus.append([q,a,label])
    return pointwise_corpus

def pairwise_data(corpus):
    '''格式化为(Q, pos_A, neg_A)的形式'''
    pairwise_corpus = dict()
    for sample in corpus:
        qid, q, aid, a, label = sample
        pairwise_corpus.setdefault(qid,dict())
        pairwise_corpus[qid].setdefault('pos',list())
        pairwise_corpus[qid].setdefault('neg',list())
        pairwise_corpus[qid]['q'] = q
        if label == 0:
            pairwise_corpus[qid]['neg'].append(a)
        else:
            pairwise_corpus[qid]['pos'].append(a)
    real_pairwise_corpus = []
    for qid in pairwise_corpus:
        q = pairwise_corpus[qid]['q']
        for pos in pairwise_corpus[qid]['pos']:
            for neg in pairwise_corpus[qid]['neg']:
                real_pairwise_corpus.append((q, pos, neg))
    return real_pairwise_corpus

def listwise_data(corpus):
    '''得到(Q, A relate to the Q)的格式'''
    listwise_data = dict()
    for sample in corpus:
        qid, q, aid, a, label = sample
        listwise_data.setdefault(qid,dict())
        listwise_data[qid].setdefault('a',list())
        listwise_data[qid]['q'] = q
        listwise_data[qid]['a'].append(a)
    real_listwise_corpus = []
    for qid in listwise_data:
        q = listwise_data[qid]['q']
        alist = listwise_data[qid]['a']
        real_listwise_corpus.append((q,alist))
    return real_listwise_corpus

if __name__ == '__main__':
    train_processed_qa = segement('WikiQA-train.tsv')
    val_processed_qa   = segement('WikiQA-dev.tsv')
    test_processed_qa  = segement('WikiQA-test.tsv')
    word2id, id2word   = build_vocab(train_processed_qa)

    transformed_train_corpus = transform(train_processed_qa,word2id)
    pointwise_train_corpus   = pointwise_data(transformed_train_corpus,keep_ids=True)
    pairwise_train_corpus    = pairwise_data(transformed_train_corpus)
    listwise_train_corpus    = listwise_data(transformed_train_corpus)

    transformed_val_corpus = transform(val_processed_qa,word2id)
    pointwise_val_corpus   = pointwise_data(transformed_val_corpus,keep_ids=True)
    pairwise_val_corpus    = pointwise_data(transformed_val_corpus,keep_ids=True)
    listwise_val_corpus    = listwise_data(transformed_val_corpus)

    transformed_test_corpus = transform(val_processed_qa,word2id)
    pointwise_test_corpus   = pointwise_data(transformed_test_corpus,keep_ids=True)
    pairwise_test_corpus    = pointwise_data(transformed_test_corpus,keep_ids=True)
    listwise_test_corpus    = listwise_data(transformed_test_corpus)

    with open(os.path.join(save_process_path,'vocab.pkl'),'w') as f:
        pickle.dump([word2id,id2word],f)
    with open(os.path.join(save_process_path,'pointwise_corpus.pkl'),'w') as f:
        pickle.dump([pointwise_train_corpus,pointwise_val_corpus],f)
