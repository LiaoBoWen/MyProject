from gensim import corpora
import jieba

dictionary = corpora.Dictionary([jieba.cut('你怎么这么可爱'),jieba.cut('我超级喜欢你')])

print(dictionary)