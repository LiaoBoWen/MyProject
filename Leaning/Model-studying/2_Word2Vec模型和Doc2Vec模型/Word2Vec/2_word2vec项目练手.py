import jieba
import warnings
# todo 在window环境使用的话 注意取消这两个警告，否则使用的时候会出现输出不稳定的情况！
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')   # 取消gensim的UserWarning
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')   # 取消gensim的FutureWarning


names = ['沙瑞金','田国富','高育良','侯亮平','钟小艾','陈岩石','欧阳菁',
         '易学习','王大路','蔡成功','孙连城','季昌明','丁义珍','郑西坡',
         '赵东来','高小琴','赵瑞龙','林华华','陆亦可','刘新建','刘庆祝']
'''

'''
for name in names:
    jieba.suggest_freq(name,True)       #todo jieba使用

with open('./temp_data/in_the_name_of_people_segment.txt',encoding='utf8') as f:
    document = f.read()

    document_cut = jieba.cut(document)

    result = ' '.join(document_cut)
    # with open('./temp_data/in_the_name_of_people_segment.txt','w',encoding='utf8') as f2:
    #     f2.write(result)

'''
## 拿到分词之后，一般的NLP任务中，需要去除停用词，由于word2vec的算法是依赖于上下文，
而上下文有可能就是停用词，因此对于word2vec我们可以不去停用词
'''

'''
## 直接读取分词后的文件到内存里面，这里使用word2vec提供的LineSentence类来读文件，
然后套用word2vec的模型，实际使用需要调参（这里省略调参步骤……）
'''

import logging
import os
from gensim.models import word2vec
from multiprocessing import cpu_count

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.ERROR)

sentences = word2vec.LineSentence('./temp_data/in_the_name_of_people_segment.txt')  # 得到的是二维的字符数组[[该句分词],[该句分词]]
# print(list(sentences))
'''
　　　　1) sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。

　　　　2) size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。

　　　　3) window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为c，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10] 8之间。

　　　　4) sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型(对低频词比较敏感)，默认是0即CBOW模型。

　　　　5) hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。

　　　　6) negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。

　　　　7) cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw,默认值也是1,不推荐修改默认值。

　　　　8) min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。

　　　　9) iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

　　　　10) alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。

　　　　11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
'''
# todo 1、hs参数调节方法   2、什么是负样本
'''
　　　架构：skip-gram（慢、对罕见字有利）vs CBOW（快）

     训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）

     欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）

     文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
'''
cores = cpu_count()
model = word2vec.Word2Vec(sentences,hs=2,min_count=1,window=2,size=200,workers=cores)
print(model.wv.vocab)

# 也可以直接索引输出对应的词向量
print('沙瑞金的词向量表示的维度 ==>',model['沙瑞金'])

#模型出来了，我们可以进行以下应用

req_count = 5


# 1、输出某一个词向量最相近的词的集合
for key in model.wv.similar_by_word('沙瑞金',topn=4):
    '''
    这里的输出不稳定
    '''
    if len(key[0]) == 3:
        req_count -= 1
        # print(key)
        print(key[0],key[1])
        if req_count == 0:
            break

# 2、输出两个词向量的相近程度(余弦相似度)
print('{}-{}==>相似度 : {} '.format('沙瑞金','高育良',model.wv.similarity('沙瑞金','高育良')))
print('{}-{}==>相似度 : {} '.format('李达康','王大路',model.wv.similarity('李达康','王大路')))


# 2.1、词汇运算(加减)  # todo 这里是负样本？
print(model.wv.most_similar(positive=['沙瑞金'],negative=['高育良'],topn=3))

# 3、找出不同类的词
print('不同类的词 : ',model.wv.doesnt_match(['高育良','沙金瑞','李达康','刘庆祝']))




# =====================================================================
# =====================================================================


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

documents = [TaggedDocument(doc,[i]) for i,doc in enumerate(common_texts)]
print(documents)
print(common_texts)
model = Doc2Vec(documents,vector_size=5,window=2,min_count=1,workers=cores)

print(model)

from gensim.test.utils import get_tmpfile

fname = get_tmpfile('my_first_doc2vec_model')
model.save(fname)
model = Doc2Vec.load(fname)

model.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)
# [ 0.09939193  0.02358485  0.07403033 -0.0598008   0.00017458]

vector = model.infer_vector(['system','response'])
print(vector)