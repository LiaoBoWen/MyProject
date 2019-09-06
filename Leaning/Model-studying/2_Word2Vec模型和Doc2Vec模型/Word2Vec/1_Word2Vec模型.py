import gensim
from gensim.models import Word2Vec,Doc2Vec
from multiprocessing import cpu_count
import numpy as np
import os

'''
## 参看我的csdn博客
'''


sentences = [['first','sentence'],
             ['second','sentence']]

'''
## Word2Vec有很多可以影响训练速度和质量的参数。第一个参数可以对字典做截断，少于min_count次数的单词会被丢弃掉, 默认值为5
model = gensim.model.Word2Vec(sentences,min_count=10)
'''
model = Word2Vec(sentences,min_count=2)
print(model['sentence'])

'''
## 另外一个是神经网络的隐藏层的单元数，推荐值为几十到几百。事实上Word2Vec参数的个数也与神经网络的隐藏层的单元数相同，
## 比如size=200，那么训练得到的Word2Vec参数个数也是200： model = Word2Vec(sentences, size=200)
## 以处理IMDB数据集为例，初始化Word2Vec对象，设置神经网络的隐藏层的单元数为200，生成的词向量的维度也与神经网络的隐藏层的单元数相同。
设置处理的窗口大小为8个单词，出现少于10次数的单词会被丢弃掉，迭代计算次数为10次，同时并发线程数与当前计算机的cpu个数相同
'''

cores = cpu_count()

model = Word2Vec(size=200,window=8,min_count=10,iter=10,workers=cores)
'''
## 创建字典并开始训练获取Word2Vec。gensim的官方文档中强调增加训练次数可以提高生成的Word2Vec的质量，
可以通过设置epochs参数来提高训练次数，默认的训练次数为5
'''
x = x_train + x_test
model.build_vocab(x)

model.train(x,total_examples=model.corpus_count,epochs=model.iter)
print(model['love'])

'''
## Word2Vec的维度与之前设置的神经网络的隐藏层的单元数相同为200，也就是说是一个长度为200的一维向量。
通过遍历一段英文，逐次获取每个单词对应的Word2Vec，连接起来就可以获得该英文段落对应的Word2Vec
'''
def getVecsByWord2Vec(model,corpus,size):
    '''
    ## 出于性能的考虑，我们将出现少于10次数的单词会被丢弃掉，
    所以存在这种情况，就是一部分单词找不到对应的Word2Vec，所以需要捕捉这个异常，通常使用python的KeyError异常捕捉即可。
    :param model:
    :param corpus:
    :param size:
    :return:
    '''
    x = []
    for text in corpus:
        xx = []
        for i, vv in enumerate(text):
            try:
                xx.append(model[vv].reshape((1,size)))
            except KeyError:
                continue
        x = np.concatenate(xx)
    x = np.array(x,dtype='float')
    return x



'''
## 以处理IMDB数据集为例，初始化Doc2Vec对象，设置神经网络的隐藏层的单元数为200，
生成的词向量的维度也与神经网络的隐藏层的单元数相同。设置处理的窗口大小为8个单词，出现少于10次数的单词会被丢弃掉，
迭代计算次数为10次，同时并发线程数与当前计算机的cpu个数相同
'''
'''
## 其中需要强调的是，dm为使用的算法，默认为1，表明使用DM算法，设置为0表明使用DBOW算法，通常使用默认配置即可
'''

max_features = 200
model = gensim.models.Doc2Vec(dm=0,dbow_word=1,size=max_features,window=8,min_count=10,iter=10,workers=cores)


'''
## 与Word2Vec不同的地方是，Doc2Vec处理的每个英文段落，需要使用一个唯一的标识标记，并且使用一种特殊定义的数据格式保存需要处理的英文段落，这种数据格式定义如下:
##其中SentimentDocument可以理解为这种格式的名称，也可以理解为这种对象的名称，words会保存英文段落，并且是以单词和符合列表的形式保存，tags就是我们说的保存的唯一标识。最简单的一种实现就是依次给每个英文段落编号，训练数据集的标记为“TRAIN_数字”，训练数据集的标记为“TEST_数字”：
'''

SentimentDocument = namedtuple('SentimentDocument', 'words tags')


def labelizeReviews(reviews,label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '{}_{}'.format(label_type,i)
        labelized.append(SentimentDocument(v,[label]))
    return labelized


x = x_train + x_test
model.build_vocab(x)
model.train(x,totla_examples=model.corpus_count,epochs=model.iter)

print(model.docvecs['I love tensorflow'])

'''
## Doc2Vec的维度与之前设置的神经网络的隐藏层的单元数相同为200，也就是说是一个长度为200的一维向量。
以英文段落为单位，通过遍历训练数据集和测试数据集，逐次获取每个英文段落对应的Doc2Vec，这里的英文段落就可以理解为数据集中针对电影的一段评价
'''
def getVecs(model,corpus,size):
    vecs = [np.array(model,docves[z.tags[0]]).reshape([1,size]) for z in corpus]
    return np.array(np.concatenate(vecs),dtype='float')

'''
## 训练Word2Vec和Doc2Vec是非常费时费力的过程，调试阶段会频繁更换分类算法以及修改分类算法参数调优，
为了提高效率，可以把之前训练得到的Word2Vec和Doc2Vec模型保存成文件形式，以Doc2Vec为例，
使用model.save函数把训练后的结果保存在本地硬盘上，运行程序时，在初始化Doc2Vec对象之前，
可以先判断本地硬盘是否存在模型文件，如果存在就直接读取模型文件初始化Doc2Vec对象，反之则需要训练数据
'''
def getDoc2vecBin(doc2vec_bin):
    if os.path.exists(doc2vec_bin):
        print('Find cache file {} '.format(doc2vec_bin))
        model = Doc2Vec.load(doc2vec_bin)
    else:
        model = Doc2Vec(size=max_features,window=5,min_count=2,workers=cores,iter=40)
        model.build_vocab(x)
        model.train(x,total_examples=model.corpus_count,epochs=model.iter)
        model.save(doc2vec_bin)