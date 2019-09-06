from multiprocessing import cpu_count
import numpy as np
import os
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import StandardScaler
from preprocession import load_file_and_split
from sklearn.svm import SVC
from sklearn.externals import joblib   #todo 保存模型


# 获取句子的所有词汇的向量，然后去平均值
# def build_word_vector(text,size,comment_w2v):   #todo 鲁棒性不错！ 但是通过对每句的词向量取平均是什么原因
#     vec = np.zeros(size).reshape((1,size))
#     count = 0
#     for word in text:
#         try:
#             vec += comment_w2v[word].reshape((1,size))
#             count += 1
#         except:
#             continue
#     if count != 0:
#         vec /= count
#     return vec


def build_word_vector(text,size,comment_w2v):
    vec = np.zeros(size).reshape([1,size])
    count = 0
    for word in text:
        try:
            vec += comment_w2v[word].reshape([1,size])
            count += 1
        except:
            continue
    if count != 0:
        vec /= count        # 这里使用直接累加的方法，会是的词之间的顺序消失，丢失部分语义
    return vec



# 训练word2vec模型
#     # 将每个词用300个维度向量化    # todo 常用的词向量维度——300维
def get_train_vecs(x_all_sentences,x_train_sentences,x_test_sentences,n_dim=300,w2v_path='w2v_model.pkl'):
    if not os.path.exists(w2v_path):
        print('训练词向量...')
        # 初始化w2v模型
        core_count = cpu_count()
        print('CPU核数:',core_count)
        comment_w2v = Word2Vec(size=n_dim,min_count=5,workers=core_count)
        # 确定词表，训练的时候吧训练的数据集和测试 的数据加起来训练
        comment_w2v.build_vocab(x_all_sentences)
        # 训练词向量
        comment_w2v.train(x_all_sentences,total_examples=comment_w2v.corpus_count,epochs=100)
        comment_w2v.save(w2v_path)
    else:
        print('加载Word2Vec...')
        # 训练数据的向量化
        comment_w2v = Word2Vec.load(w2v_path)
    train_vectors = np.concatenate([build_word_vector(z,n_dim,comment_w2v) for z in x_train_sentences])
    test_vectors = np.concatenate([build_word_vector(z,n_dim,comment_w2v) for z in x_test_sentences])

    return train_vectors, test_vectors


# def get_train_vecs(x,train,test):
#     n_dim = 300
#     comment_w2v = Word2Vec(size=n_dim,min_count=5)
#     comment_w2v.build_vocab(x)
#     comment_w2v.train(x,total_examples=comment_w2v.corpus_count,epochs=100)
#     comment_w2v.save('w2v_model.pkl')
#     train_vectors = np.concatenate()


# 训练SVM模型做分类器
def svm_train(train_vecs,y_train,test_vecs,y_test,model_path='svm_model.pkl'):
    if not os.path.exists(model_path):
        print('训练svm模型')
        standardScaler = StandardScaler()
        standardScaler.fit(train_vecs)

        train_vecs =  standardScaler.transform(train_vecs)
        test_vecs  = standardScaler.transform(test_vecs)

        clf = SVC(kernel='rbf',verbose=True)
        clf.fit(train_vecs,y_train)
        joblib.dump(clf,model_path)
    else:
        print('加载svm模型')
        clf = joblib.load(model_path)
    print('svm模型准确率：{:.6%}'.format(clf.score(test_vecs,y_test)))


if __name__ == "__main__":
    x,x_train,x_test,y_train,y_test = load_file_and_split()
    train_vec,test_vec = get_train_vecs(x,x_train,x_test)
    print(train_vec)
    svm_train(train_vec,y_train,test_vec,y_test)