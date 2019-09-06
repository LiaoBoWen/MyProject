import numpy as np
from collections import Counter

def poetry_process(batch_size=64,poetry_file='./data/poems.txt'):
    poetrys = []
    # 1、对诗的长度进行选择，选择不含不合理符号的诗
    with open(poetry_file,'r',encoding='utf') as f:
        for line in f:
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ','')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                content_len = len(content)
                if content_len < 5 or content_len > 79:
                    continue
                content = 'B' + content + 'E' # 给每首诗开头和结束做标记
                poetrys.append(content)
            except Exception as e:
                print(e)

    # 2、按照诗的长度进行排序
    # poetrys = sorted(poetrys,key=lambda x :len(x))
    print('诗词数量：',len(poetrys))

    # 3、统计字出现次数
    all_word = []
    for poetry in poetrys:
        all_word += [word for word in poetry]
    counter = Counter(all_word)
    del all_word

    count_pairs = sorted(counter.items(),key=lambda x : -x[1])
    words, _ = zip(*count_pairs)

    words = words[:len(words)] + (' ',)

    # 4、使用zip巧妙的生成单词字典
    word_num_map = dict(zip(words,range(len(words))))

    # 5、诗词转换为向量
    to_num = lambda word: word_num_map.get(word,len(words))
    poetry_vector = [list(map(to_num,poetry)) for poetry in poetrys]

    n_check = len(poetry_vector) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_check):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batches = poetry_vector[start_index:end_index]
        length = max(map(len,batches))
        xdata = np.full([batch_size,length],word_num_map[' '],np.int32)
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row]
        ydata = np.copy(xdata)
        # y 就是x接下来一个字
        ydata[:,:-1] = xdata[:,1:]
        '''
           xdata             ydata
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
        '''
        x_batches.append(xdata)
        y_batches.append(ydata)
    return words, poetry_vector, to_num, x_batches, y_batches