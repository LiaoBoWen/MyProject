import os
import re
import jieba
import tensorflow as tf
from collections import Counter

from Hyparams import hyparams as hp


def remove_punc(sent):
    sent = re.sub(r'[,.!@#$%^&*()+_=！￥:"\'‘’“”\n]','',sent)
    return sent

def get_data(path='./data'):
    with open(path + '/xiaohuangji50w_nofenchi.conv','r') as dataset:
        sentences = dataset.readlines()

        source = []
        target = []

        for i in range(1,len(sentences),2):
            if '小通' not in sentences[i] and '小通' not in sentences[i + 1]:
                source.append(remove_punc(sentences[i]))
                target.append(remove_punc(sentences[i+1]))

    return source, target

def generator_vocab(source, target, top_n=20000-4):
    '''去除top_n词'''
    vocab = Counter()

    for sent in source:
        vocab.update(jieba.cut(sent))
    for sent in target:
        vocab.update(jieba.cut(sent))

    # 去除英文和数字
    vocab_keys = list(vocab.keys())
    for word in vocab_keys:
        if word.encode('utf8').isalnum():
            vocab.pop(word)

    vocab  = vocab.most_common(top_n)   # [(W,n),(,),(,)...]

    return vocab

def save_vocab(vocab,path='./data'):
    with open(path + '/vocab.txt','w') as f:
        f.write('<PAD>\n')
        f.write('<UNK>\n')
        f.write('<S>\n')
        f.write('</S>\n')

        f.writelines([word[0] + '\n' for word in vocab])

def load_vocab(path='./data/vocab.txt'):
    with open(path,'r') as f:
        vocab = f.read().splitlines()
        idx2word = dict(enumerate(vocab))
        word2idx = dict(zip(idx2word.keys(),idx2word.values()))
    return idx2word, word2idx





# 以下部分是数据迭代部分所用到的函数
def pad(data,length,padding_sign):
    new_data = data
    pad_len = length - len(new_data)
    for _ in range(pad_len):
        new_data.append(padding_sign)
    return new_data

def generate_fn(sources, targets, vocab_path):
    word2idx, idx2word = load_vocab(vocab_path)
    for source, target in zip(sources, targets):
        x = []
        y = [2]

        for word in source:
            x.append(word2idx.get(word,1))

        if len(x) >= hp.max_len:
            x = x[:hp.max_len]
            x[-1] = 3
        else:
            x = pad(x,hp.max_len,0)

        for word in targets:
            y.append(word2idx.get(word,1))

        if len(y) >= hp.max_len:
            y = y[:hp.max_len]
            y[-1] = 3
        else:
            y = pad(y,hp.max_len,1)

        y_input, y = y[:-1], y[1:]
        yield (x, len(x), source), (y_input, y, len(y), target)

def input_fn(sources, targets, vocab_path, batch_size,shuffle=False):
    shapes = (
        ([None],(),()),
        ([None],[None],(),())
    )
    types = (
        (tf.int32,tf.int32,tf.string),
        (tf.int32,tf.int32,tf.int32,tf.string)
    )

    paddings = ((0,0,''),
                (0,0,0,''))

    dataset = tf.data.Dataset.from_generator(
        lambda : generate_fn(sources, targets, vocab_path),
        output_shapes=shapes,
        output_types=types,
    )

    dataset.repeat()  # iterator forever...
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(path,vocab_path,batch_size=hp.batch_size,shuffle=False):
    sources, targets = get_data(path)           # 这里为什么是get_data ？？ 都不用分词的吗？？？？
    batches = input_fn(sources, targets, vocab_path, batch_size,shuffle)
    num_batches = calc_num_batches(len(sources),batch_size)
    return batches, num_batches, len(sources)


def remove_punc(sent):
    sent = re.sub(r'[,.;\'"“”’‘《》？！!@#￥%$]','',sent)
    return sent


def get_data(path):
    with open(path) as f:
        sentences = f.readlines()
        sources = []
        targets = []

        for i in range(1,len(sentences),2):
            if '小通' not in sentences[i] and '小通' not in sentences[i + 1] :
                sources.append(remove_punc(sentences[i]))
                targets.append(remove_punc(sentences[i + 1]))
        return sources ,targets

def pad(inputs,length,pad_sign):
    input_len = len(inputs)
    pad_len = length - input_len
    for _ in range(pad_len):
        inputs.append(pad_sign)
    return inputs

def generate_fn(sources,targets,vocab_path):
    word2idx, idx2word = load_vocab(vocab_path)
    x = []
    y = [2]

    for source, target in zip(sources, targets):
        for word in source :
            x.append(word2idx.get(word,1))

        if len(x) >= hp.max_len:
            x = x[:20]
            x[-1] = 3
        else:
            x = pad(x,hp.max_len,word2idx['<PAD>'])

        for word in target:
            y.append(word2idx.get(word,1))

        if len(y) > hp.max_len:
            y = y[:20]
            y[-1] = 3

        y_input, y = y[:-1], y[1:]

        yield (x,len(x),source), (y_input,y,len(y),target)

def input_fn(sources,targets,vocab_path,batch_size,shuffle=False):
    shapes = (
        ([None],(),()),
        ([None],[None],(),())
    )

    types = (
        (tf.int32,tf.int32,tf.string),
        (tf.int32,tf.int32,tf.int32,tf.string)
    )

    paddings = (
        (0,0,''),
        (0,0,0,'')
    )

    dataset = tf.data.Dataset.from_generator(
        lambda : generate_fn(sources, targets,vocab_path),
        output_types=types,
        output_shapes=shapes
    )
    dataset.repeat()
    dataset = dataset.padded_batch(batch_size,padded_shapes=shapes,padding_values=paddings)

    return dataset

def get_batch(path,vocab_path,batch_size=hp.max_len,shuffle=False):
    sources, targets = get_data(path)
    batches = input_fn(sources,targets,vocab_path=vocab_path,batch_size=batch_size,shuffle=shuffle)
    num_batches = (len(sources) // batch_size) + int(len(sources) % batch_size != 0)
    return batches, num_batches, len(sources)


def get_batch_eval(query, vocab_path):
    target = ''
    batches = input_fn(query,target,vocab_path,1,False)
    return batches