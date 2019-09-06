'''
使用的gensim的LDA模型  word-bow => tfidf => lsi(lda) => sim
'''
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models, similarities

train = []

with open('data/lda_logfile/stopwords.txt',encoding='utf8') as f:
    stopwords = [i.strip() for i in f]

documents = ["Shipment of gold damaged in a fire",
            "Delivery of silver arrived in a silver truck",
            "Shipment of gold arrived in a truck"]

texts = [[word for word in document.lower().split()] for document in documents]

# 词袋
print('=====bag-of-words=====')
dictionary = corpora.Dictionary(texts)  # 这里的输入的是句子切割后的列表形式存在的句子
print(dictionary)
print(dictionary.token2id)      # {'a': 0, 'damaged': 1, 'fire': 2, 'gold': 3, 'in': 4, 'of': 5, 'shipment': 6, 'arrived': 7, 'delivery': 8, 'silver': 9, 'truck': 10}

print('====doc2bow====')
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)   # 这就是真正的词袋：[[(字典对应的value，本句出现次数) * count(setence's words)]]      # [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(0, 1), (4, 1), (5, 1), (7, 1), (8, 1), (9, 2), (10, 1)], [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (10, 1)]]




# 计算TF-IDF模型
print('====TF-IDF模型====')
tfidf = models.TfidfModel(corpus,)
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)      # [(1, 0.6633689723434505), (2, 0.6633689723434505), (3, 0.2448297500958463), (6, 0.2448297500958463)]
                    # [(7, 0.16073253746956623), (8, 0.4355066251613605), (9, 0.871013250322721), (10, 0.16073253746956623)]

# 发现有一些token丢失了,查看：
print(tfidf.dfs)    #{0: 3, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 2, 7: 2, 8: 1, 9: 1, 10: 2}

print(tfidf.idfs)      # {0: 0.0, 1: 1.5849625007211563, 2: 1.5849625007211563, 3: 0.5849625007211562, 4: 0.0, 5: 0.0, 6: 0.5849625007211562, 7: 0.5849625007211562, 8: 1.5849625007211563, 9: 1.5849625007211563, 10: 0.5849625007211562}

# 0， 4， 5这3个单词的文档数（df)为3，而文档总数也为3，所以idf被计算为0了，看来gensim没有对分子加1，做一个平滑。不过我们同时也发现这3个单词分别为a, in, of这样的介词，完全可以在预处理时作为停用词干掉，这也从另一个方面说明TF-IDF的有效性。

# 训练LSI模型   # lsi的物理意义不太好解释，不过最核心的意义是将训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解
print('====LSI模型====')
lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)
print(lsi.print_topics(2)) # 这个做man解释 打印主题

corpus_lsi = lsi[corpus_tfidf]
for doc in corpus_lsi:
    print(doc)

print('====LDA模型====')
lda = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=3)
print(lda.print_topics(3))

corpus_lda = lda[corpus_tfidf]
for doc in corpus_lda:
    print(doc)


# 计算（余弦）相似度
index =  similarities.MatrixSimilarity(lsi[corpus])


############ 例子 #############
print('====例子====')
query = 'gold silver truck'
query_bow = dictionary.doc2bow(query.lower().split())
print(query_bow)

query_lsi = lsi[query_bow]  # 判断属于哪类主题
print(query_lsi)

sims = index[query_lsi]  #  得到的和三类的相似度
print(list(enumerate(sims))) # 得到的和三类的相似度