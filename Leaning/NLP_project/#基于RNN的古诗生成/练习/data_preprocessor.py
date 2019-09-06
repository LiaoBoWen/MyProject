import numpy as np

from collections import Counter
import pickle
import os

def poetry_process(poem_path='../data/poems.txt'):
    #1.数据的清理和选择
    poems = []
    with open(poem_path,'r',encoding='utf8 ') as f:
        for line in f.readlines():
            # 最后一行是空的，切分会报错，跳过这个错误
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ','')
                if '_' in content or '<' in content or '《' in content or '(' in content or '（' in content:
                    continue
                content_len = len(content)
                if content_len < 5 or content_len > 79:
                    continue
                poems.append('B' + content + 'E')
            except:
                pass
    # 2. 字典化
    all_words = ''.join(poems)
    counter = Counter(all_words)
    del all_words

    count_pair = sorted(counter.items(),key=lambda x:-x[1])
    words, _ = zip(*count_pair)
    words = words + (' ',)
    word_num_map = dict(zip(words,range(len(words))))
    to_num = lambda word: word_num_map.get(word,len(word_num_map))


    poetry_vector = [list(map(to_num,poem)) for poem in poems]    # good idea

    return words, poetry_vector, to_num, word_num_map

        # 3.批量处理
def get_batches(poetry_vector,word_num_map,batch_size=64):
    # poetry_vector = pickle.load(open('./model/poetry_vector.pkl','rb'))
    n_check = len(poetry_vector) // batch_size # 这杨不会出现数据溢出的情况

    # word_num_map = pickle.load(open('./model/word_num_map.pkl','rb'))


    for i in range(n_check):
        start_index = i * batch_size
        end_index = i * batch_size + batch_size
        batches = poetry_vector[start_index:end_index]
        length_max = max(list(map(len,batches)))
        x_data = np.full([batch_size,length_max],word_num_map[' '],np.int32)
        for row in range(batch_size):
            x_data[row,:len(batches[row])] = batches[row]
        y_data = np.copy(x_data)    # 需要注意地址问题
        y_data[:,:-1] = x_data[:,1:]

        yield x_data, y_data    # 使用生成器减少压力