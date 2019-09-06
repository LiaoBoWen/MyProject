from sklearn.feature_extraction.text import CountVectorizer

'''

===================================
 CountVectorizer参数介绍
===================================
############ 可以自动过滤中英文标点符号
## decode_error，处理解码失败的方式，分为‘strict’、‘ignore’、‘replace’三种方式。
## strip_accents，在预处理步骤中移除重音的方式。
## max_features，词袋特征个数的最大值。
## stop_words，判断word结束的方式。
## max_df，df（文档词频）最大值。
## min_df，df最小值。
## binary，默认为False，当与TF-IDF结合使用时需要设置为True。 本例中处理的数据集均为英文，所以针对解码失败直接忽略，使用ignore方式，stop_words的方式使用english，strip_accents方式为ascii方式。
'''
vectorizer = CountVectorizer(min_df=1)

corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one',
          'Is this the first document?']

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())
'''
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
[[0 1 1 1 0 0 1 0 1]
 [0 1 0 1 0 2 1 0 1]
 [1 0 0 0 1 0 1 1 0]
 [0 1 1 1 0 0 1 0 1]]

'''


# todo 也可以使用现有的词袋的特征对其他的文本进行特和提取
vocabulary = vectorizer.vocabulary_
# print(vocabulary)
'''
{'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
'''
#todo 注意！当vocabulary参数不为空的话，min_df失效！！！
new_vectorizer = CountVectorizer(min_df=3,vocabulary=vocabulary)
corpus_ = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document ?',
           'are you ok first?']
test = new_vectorizer.fit_transform(corpus_)
print(test)
