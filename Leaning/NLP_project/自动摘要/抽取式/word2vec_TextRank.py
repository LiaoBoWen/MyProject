import os
import re
import jieba
import networkx as nx
import warnings
from multiprocessing import cpu_count
import numpy as np
warnings.filterwarnings(action='ignore',category=UserWarning,module='smart_open')

from gensim.models import Word2Vec,word2vec

def get_stopword(stop_words_path='./data/stopwords.txt'):
    with open(stop_words_path,'r',encoding='utf8') as f:
        stop_words = f.read().splitlines()
    return stop_words

def data_process(data_paths='./data/cnews/news_{}.txt'):
    stop_words = get_stopword()

    with open(data_paths.format(1),'r',encoding='utf8') as f:
        text_1 = f.read()
        text_1 = re.sub(r'\n','',text_1)
        text_1 = re.sub(r'\d+','',text_1)
    with open(data_paths.format(2),'r',encoding='utf8') as f:
        text_2 = f.read()
        text_2 = re.sub(r'\n','',text_2)
        text_2 = re.sub(r'\d+','',text_2)
    with open(data_paths.format(3),'r',encoding='utf8') as f:
        text_3 = f.read()
        text_3 = re.sub(r'\n','',text_3)
        text_3 = re.sub(r'\d+','',text_3)

    text = text_1 + text_2 + text_3

    jieba.suggest_freq(['易会满'])
    processed_text = ' '.join([word for word in jieba.cut(text) if word not in stop_words])
    with open('./data/processed_txt','w',encoding='utf8') as f:
        f.write(processed_text)

    processed_text = word2vec.LineSentence('./data/processed_txt')

    return processed_text



def train_word2vec(data,model_path='./model/w2v.pkl',embedding_size=200):
    print('Training w2v model ...')
    w2v = word2vec.Word2Vec(data,hs=1,size=embedding_size,window=5,min_count=1,workers=cpu_count(),iter=300)
    # w2v.build_vocab(data)
    # w2v.train(data,total_examples=w2v.corpus_count,epochs=128)
    w2v.save(model_path)
    print('Saved w2v_model !')

    return w2v


def get_word2vec(model_path='./model/w2v.pkl',embedding_size=300):
    if not os.path.exists('./model'):
        os.makedirs('./model')

    if os.path.exists(model_path):
        print('Loading w2v model ...')
        w2v = Word2Vec.load(model_path)
        print('Loaded !')
    else:
        processed_data = data_process()
        # print(processed_data)
        w2v = train_word2vec(processed_data,embedding_size=embedding_size)

    return w2v


def cut_sentences(datas_path='./data/cnews'):
    all_sentences = []

    files = [os.path.join(datas_path,file) for file in os.listdir(datas_path)]
    for file in files:
        with open(file,'r',encoding='utf8') as f:
            for line in f.readlines():
                if line.strip():
                    sentences = re.split(r'[；。！？]',line)
                    sentences = [line.strip() for line in sentences if len(line.strip()) > 1]
                    all_sentences.extend(sentences)
    return all_sentences

def seg_sentence(sentences,stop_words):
    all_sentences = []
    jieba.suggest_freq('易会满',True)
    for sentence in sentences:
        sentence = re.sub(r'\d+','',sentence)
        sentence = [word for word in jieba.cut(sentence) if word not in stop_words and word != ' ']
        all_sentences.append(sentence)

    return all_sentences

def embedding(sentences,w2v):
    vec = []
    for sentence in sentences:
        if len(sentence) == 0:
            vec.append(np.zeros(300))
            continue
        vec_ = []
        for word in sentence:
            try:
                vec_.append(w2v[word])
            except:
                vec_.append(np.zeros(300))
        vec.append(np.mean(vec_,axis=0))
    return np.reshape(vec,[-1,300])

def cosine_similar(vecs):
    len_vecs = len(vecs)
    sim = vecs.dot(vecs.T) / (np.sqrt(np.sum(np.power(vecs,2),axis=-1).reshape(-1,1)).dot(np.sqrt(np.sum(np.power(vecs,2),axis=-1).reshape(1,-1))) + 1e-16)
    for _ in range(len_vecs):
        # 为了pagerank确推理，需要把对角的相似度（自身的相似度）去掉
        sim[_][_] = 0

    return sim

def make_corpus(sim_mat,topn=10):
    # 利用句子相似度构建图结构，句子作为节点，相似度作为转移概率
    nx_graph = nx.from_numpy_array(sim_mat)

    # 得到句子的textrank
    scores = nx.pagerank(nx_graph)
    # print(scores)

    ranked_sentences = sorted([(scores[i],s) for i, s in enumerate(sentences)],reverse=True)

    for i in range(topn):
        print('[{}]: {}\n'.format(i + 1,ranked_sentences[i][1]))


if __name__ == '__main__':
    w2v_model = get_word2vec()

    stop_words = get_stopword()
    sentences = cut_sentences()
    processed_sentences = seg_sentence(sentences,stop_words)
    # print(processed_sentences)

    vecs = embedding(processed_sentences,w2v_model)
    # print(w2v_model.most_similar(positive=['配资'],negative=['理事长']))
    # print(vecs)

    sim_mat = cosine_similar(vecs)
    # print(sim_mat)

    make_corpus(sim_mat)