from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)

'''
tf-idf模型通常和词袋模型配合使用，对词袋模型生成的数组进一步处理
当然也可以直接使用CountVectorizer和TfidfTransformer的结合——TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer()
result = tfidf2.fit_transform(corpus)
'''

counts = [[3,0,1],
          [2,0,0],
          [3,0,0],
          [4,0,0],
          [3,2,0],
          [3,0,2]]
tfidf = transformer.fit_transform(counts)
# 这里的TF-IDF没有使用平滑
print(tfidf.toarray())
'''
[[0.81940995 0.         0.57320793]
 [1.         0.         0.        ]
 [1.         0.         0.        ]
 [1.         0.         0.        ]
 [0.47330339 0.88089948 0.        ]
 [0.58149261 0.         0.81355169]]
'''

# 特征提取还可以使用词袋模型的加强版n-gram，比如最常见的2-gram，这样可以更好的提取单词前后之间的关系。
corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one',
          'Is this the first document?']
corpus = ['你 好 吗 ？',
          '我 很 好 呀。',
          '那 就 这样 吧， 再见。。。\n',
          '\n']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer(ngram_range=(1,2),token_pattern=r"(?u)\b\w+\b") # 由于token_patten参数的原因所以单个字的会被过滤，因为按照英文来说一个字母的单吃无关紧要
result = tfidf2.fit_transform(corpus)
print(tfidf2.get_feature_names())
print(tfidf2.vocabulary_)
print(result)
print(result.nonzero()[0])
print(result.toarray())
print(result.todense())
'''
['and', 'and the', 'document', 'first', 'first document', 'is', 'is the', 'is this', 'one', 'second', 'second document', 'second second', 'the', 'the first', 'the second', 'the third', 'third', 'third one', 'this', 'this is', 'this the']
{'this': 18, 'is': 5, 'the': 12, 'first': 3, 'document': 2, 'this is': 19, 'is the': 6, 'the first': 13, 'first document': 4, 'second': 9, 'the second': 14, 'second second': 11, 'second document': 10, 'and': 0, 'third': 16, 'one': 8, 'and the': 1, 'the third': 15, 'third one': 17, 'is this': 7, 'this the': 20}
  (0, 18)	0.29752161385065906
  (0, 5)	0.29752161385065906
  (0, 12)	0.24324341450436152
  (0, 3)	0.36749838344552144
  (0, 2)	0.29752161385065906
  (0, 19)	0.36749838344552144
  (0, 6)	0.36749838344552144
  (0, 13)	0.36749838344552144
  (0, 4)	0.36749838344552144
  (1, 18)	0.20454414683286498
  (第一行，列表的下标)
  
  
  
[[0.         0.43877674 0.54197657 0.4387760.         0.
  0.35872874 0.         0.43877674]
 [0.         0.27230147 0.         0.27230147 0.         0.85322574
  0.22262429 0.         0.27230147]
 [0.55280532 0.         0.         0.         0.55280532 0.
  0.28847675 0.55280532 0.        ]
 [0.         0.43877674 0.54197657 0.43877674 0.         0.
  0.35872874 0.         0.43877674]]
'''