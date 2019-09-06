# ======================================================================================
# 在gensim下使用fattext方法
# ======================================================================================
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')

from gensim.models import FastText
from gensim.test.utils import common_texts

model = FastText(common_texts,size=4,window=3,min_count=1,iter=10,)
print(common_texts)


from gensim.test.utils import get_tmpfile

fname = get_tmpfile('fasttext.model')
model.save(fname)
model = FastText.load(fname)

existent_word = 'computer'
print(existent_word in model.wv.vocab)
# todo 这里需要注意，由于FastText的n-gram机制所以的现象
existent_word_test = '电脑'
print(existent_word_test in model)


# 像word2vec一样去除向量表示
computer_vec = model[existent_word]
print(computer_vec)
print(model.wv[existent_word])

# todo 对于未收录的词也可以得到训练的词向量，可靠性未知
oov_word = 'graph-out-of-vocab'
print(oov_word in model.wv.vocab)
print(model.wv[oov_word])
print(model.wv['graph-computer'])

similarities = model.wv.most_similar(positive=['computer','human'],negative=['interface'])
most_similar = similarities[0]

similarities = model.wv.most_similar_cosmul(positive=['computer','human'],negative=['interface'])

not_matching = model.wv.doesnt_match('human computer interface tree'.split())
print('not match ',not_matching)

sim_score = model.wv.similarity('computer','human')

from gensim.test.utils import datapath
# similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
analogies_result = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
print(analogies_result)

# todo 由于FastText的n-gram的机制，对于ngram的字符出现在training data的话也是可以进行表示的
print(model.wv.similarity('computer','computers'))


