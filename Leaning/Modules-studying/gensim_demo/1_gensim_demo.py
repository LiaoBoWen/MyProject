import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')
from gensim import corpora
from gensim.test.utils import common_texts

print('输入数据：',common_texts)

# 类似于字典化的表示词袋模型，应对大型稀疏矩阵的占用问题
dictionary = corpora.Dictionary(common_texts)
# 把单词进行bow词袋化
corpus = [dictionary.doc2bow(text) for text in common_texts]
print(corpus)

from gensim.models import TfidfModel

# 生成的时迭代类型应对内存问题
tfidf = TfidfModel(corpus)
print(tfidf[[0,1],[1,1],[2,1]])