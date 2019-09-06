import jieba.posseg as posg
import os
import pickle
from gensim import corpora, models, similarities # corpora是什么模块
from data_util import tokenizer


def generate_dic_and_corpus(knowledge_file,file_name,stop_words):
    knowledge_texts = tokenizer(knowledge_file,stop_words)
    train_texts = tokenizer(file_name,stop_words)

    # 保存字典
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    dictionary = corpora.Dictionary(knowledge_texts + train_texts)
    dictionary.save(os.path.join('./tmp/dictionary.dict'))

    corpus = [dictionary.doc2bow(text) for text in knowledge_texts] # corpus of knowledge
    corpora.MmCorpus.serialize('./tmp/knowledge_corpus.mm',corpus)    # todo 啥方法????

def topK_sim_ix(file_name,stop_words,K):
    sim_path = 'tmp/' + file_name[5:-4]
    if os.path.exists(sim_path):
        with open(sim_path,'rb') as f:
            sim_ixs = pickle.load(f)
        return sim_ixs

    # load dictionary and corpus
    dictionary = corpora.Dictionary.load('tmp/dictionary.dict')
    corpus = corpora.MmCorpus('tmp/knowledge_corpus.mm')

    lsi = models.LsiModel(corpus,id2word=dictionary,num_topics=10)

    index = similarities.MatrixSimilarity(lsi[corpus])
    sim_ixs = []
    with open(file_name,encoding='utf8') as f:
        tmp = []
        for i, line in enumerate(f):
            if i % 6 == 0:
                tmp.extend([token for token, _ in posg.cut(line.rstrip()) if token not in stop_words])
            if i % 6 == 1:
                tmp.extend([token for token, _ in posg.cut(line.rstrip()) if token not in stop_words])
                vec_lsi = lsi[dictionary.doc2bow(tmp)]
                sim_ix = index[vec_lsi]
                sim_ix = [i for i, j in sorted(enumerate(sim_ix),key=lambda item:-item[1])[:K]]
                sim_ixs.append(sim_ix)
                tmp.clear()
        with open(sim_path,'wb') as f:
            pickle.dump(sim_ixs,f)
        return sim_ixs