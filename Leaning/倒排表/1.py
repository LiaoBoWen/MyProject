import jieba
from collections import defaultdict

list_inverted_index = []
inverted_index_dict = {}
count_dict = {}

corpus = ['我真的爱你,你知不知道','你呢']

for i in range(len(corpus)):
    list_word = []
    for word in jieba.cut(corpus[i]):
        if word not in list_inverted_index:
            list_inverted_index.append(word)
            inverted_index_dict[word] = {i:1}
            list_word.append(word)
        elif word not in list_word:
            inverted_index_dict[word][i] = 1
            list_word.append(word)
        else:
            inverted_index_dict[word][i] += 1

print(list_inverted_index)
print(inverted_index_dict)  # {key:{出现在的文章id:在该文出现次数}}