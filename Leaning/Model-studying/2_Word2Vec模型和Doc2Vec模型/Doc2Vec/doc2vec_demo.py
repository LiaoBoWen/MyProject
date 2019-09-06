import os
import sys
import jieba
import numpy as np
from jieba import analyse  # 提取关键词
from multiprocessing import cpu_count
from tqdm import tqdm

from sklearn import utils

from gensim.models.doc2vec import Doc2Vec, LabeledSentence, TaggedDocument


def get_dataset():
    with open('./data/questions.txt',encoding='utf8') as f:
        corpus = f.read().splitlines()

        # 添加自定义的词库用于切分或重组模板不能处理的词组
        jieba.load_userdict('./data/userdict.txt')

        stop_words = set(open('./data/stopwords.txt',encoding='utf8').read().strip().splitlines())

        text = []

        for i, sentence in enumerate(corpus):
            sentence_ = ' '.join([w for w in jieba.cut(sentence) if w not in stop_words])
            print(TaggedDocument(sentence_,tags=[i]))
            text.append(TaggedDocument(sentence_,tags=[i]))

    return text


def get_vec(model,text,size=200):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1,size)) for z in text]

    return np.concatenate(vecs)


def train(corpus,size=200,epoch_num=1):
    model = Doc2Vec(corpus,min_count=1,window=5,size=size,sample=1e-3,negative=5,workers=cpu_count(),hs=1,iter=6)

    model.train(corpus,total_examples=model.corpus_count,epochs=70)

    model.save('doc2vecModel')

    # print(get_vec(model,get_dataset()))

    return model

def test():
    model = Doc2Vec.load('doc2vecModel')

    test_ = '申请贷款需要什么条件？'

    stop_words = set(open('./data/stopwords.txt',encoding='utf8').read().splitlines())

    test1 = [w for w in jieba.cut(test_) if w not in stop_words]

    # 获取输入句子对应的向量
    inference_vector = model.infer_vector(doc_words=test1)
    # print(inference_vector)

    # 返回相似的句子
    sims = model.docvecs.most_similar([inference_vector],topn=2)

    return sims



def train_feature(text):
    model = Doc2Vec(size=200,negative=5,hs=0,min_count=2,sample=0,workers=cpu_count())

    model.build_vocab([x for x in tqdm(text)])

    for _ in range(30):
        model.train(utils.shuffle([x for x in text]),total_examples=model.corpus_count,epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model

def vec_for_learning(model,docs):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in docs])

    return targets, regressors


if __name__ == '__found similarity__':
    text = get_dataset()

    train(text)

    sim = test()
    print(sim)


if __name__ == '__main__':
    text = get_dataset()

    model = train_feature(text)

    targets, regresses = vec_for_learning(model,text)

    print(targets[0])

    print(regresses[0])

