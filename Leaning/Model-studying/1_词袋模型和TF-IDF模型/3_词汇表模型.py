import tensorflow as tf
from tensorflow.contrib import learn

'''
##############################################
词袋模型可以很好的表现文本由哪些单词组成，
但是却无法表达出单词之间的前后关系，
于是人们借鉴了词袋模型的思想，
使用生成的词汇表对原有句子按照单词逐个进行编码。
TensorFlow默认支持了这种模型：
##############################################



## max_document_length:，文档的最大长度。如果文本的长度(单词长度)大于最大长度，那么它会被剪切，反之则用0填充。
## min_frequency，词频的最小值，出现次数小于最小词频则不会被收录到词表中。
## vocabulary，CategoricalVocabulary 对象。
## tokenizer_fn，分词函数。
'''

x_text = [
    'i love you.',
    'me,too.'
]

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=4)
vocab_processor.fit(x_text)
print(next(vocab_processor.transform(x_text)))
print(next(vocab_processor.transform(['i,,,,me ,too'])))
'''
[1 2 3 0]
-----------------------
[1 4 5 0]
'''