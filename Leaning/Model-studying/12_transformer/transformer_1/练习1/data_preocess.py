from hypeParams import params
from utils import *
from collections import Counter
import jieba
import re

def remove_punc(line):
    line = re.sub(r'[,.?!@#$%^&*)(，。、！\'"’“‘”]','',line)
    return line

def generate_dataset(path='../data'):
    '''
    区分Q、A
    '''
    with open(path + '/xiaohuangji50w_nofenci.conv',encoding='utf8') as dataset:
        sentence = []

        for i in dataset:
            sentence.append(i)

        sources = []
        targets = []

        for line in range(len(sentence)):
            if sentence[line][0] == 'E':
                if '小通' not in sentence[line + 1] and '小通' not in sentence[line + 2]:
                    sources.append(remove_punc(sentence[line + 1][2:-1]))
                    targets.append(remove_punc(sentence[line + 2][2:-1]))

    return sources, targets

def generate_vocab(source,target,number_words=20000):
    vocab = Counter()

    for sent in source:
        vocab.update(Counter(jieba.cut(sent)))
    for sent in target:
        vocab.update(Counter(jieba.cut(sent)))

    vocab = vocab.most_common(number_words) # 这里使用的list形式

    return vocab

def save_vocab(vocab,path='./data'):
    '''
    为了不用每次都要计算频率，所以保存vocab文件
    '''
    vocab_keys = [w[0] + '\n' for w in vocab]  # 为了方便写入这里加入\n

    with open(path + '/vocab.txt','w',encoding='utf8') as file:
        file.write('<PAD>\n')
        file.write('<UNK>\n')
        file.write('<S>\n')
        file.write('</S>\n')
        file.writelines(vocab_keys)

def load_vocab(vocab_path='./data/vocab.txt'):
    '''
    word2index && index2word
    '''
    with open(vocab_path,'r',encoding='utf8') as file:
        vocab = [w for w in file.read().splitlines()]
        word2index = {word:index for index, word in enumerate(vocab)}
        index2word = {index:word for index, word in enumerate(vocab)}

    return word2index, index2word

def pad(data,length,word2index):
    '''
    对每句进行padding
    '''
    num_pad = length - len(data)
    for i in range(num_pad):
        data.append(word2index.get('<PAD>'))
    return data

def generate_fn(sources, targets, vocab_path):
    '''get source target after pad&&cut'''
    word2index, index2word = load_vocab(vocab_path)
    for source, target in zip(sources,targets):
        x = []
        y = [word2index.get('<S>')]

        for word in jieba.cut(source):
            x.append(word2index.get(word,word2index['<UNK>']))
        x.append(word2index['</S>'])

        if len(x) > params.max_len:
            x = x[:params.max_len]
            x[-1] = word2index['</S>']
        else:
            x = pad(x,params.max_len,word2index)

        for word in jieba.cut(target):
            y.append(word2index.get(word,word2index['<UNK>']))
        y.append(word2index['</S>'])

        if len(y) > params.max_len:
            y = y[:params.max_len]
            y[-1] = word2index['</S>']
        else:
            y = pad(y,params.max_len,word2index)

        decoder_input, y = y[:-1], y[1:]

        yield (x,len(x),source), (decoder_input,y,len(y),target)    # 这里和机器翻译的target部分一样

def input_fn(sources,targets,vocab_path,batch_size,shuffle=False):
    #batch ??? todo two new functions
    shapes = (([None],(),()),
             [None],[None],(),())
    types = ((tf.int32,tf.int32,tf.string),
            (tf.int32,tf.int32,tf.int32,tf.string))
    paddings = ((0,0,''),
               (0,0,0,''))
    dataset = tf.data.Dataset.from_generator(
        generate_fn,
        output_types=types,
        output_shapes=shapes,
        args=(sources,targets,vocab_path)   # 这里arguments是传递给generate_fn的参数
    )

    dataset = dataset.repeat() # iterator forever
    dataset = dataset.padded_batch(batch_size,shapes,paddings).prefetch(1)   # todo new function

    return dataset

def get_batch(path,vocab_path,batch_size=params.max_len,shuffle=False):  # todo 这里的两个path一样 ？
    sources, targets = generate_dataset(path)
    batches = input_fn(sources,targets,vocab_path,batch_size,shuffle=shuffle)
    num_batches = calc_num_batch(len(sources),batch_size)
    return batches, num_batches, len(sources)

def get_batch_evaluate(query,vocab_path):
    # 测试部分
    targets = ''
    batch = input_fn(query,targets,vocab_path,1,shuffle=False)
    return batch