from Hyperparams import hyperparams as hp
from utils import *
import jieba
import numpy as np

# remove punctuations
def remove_punc(line):
    line = line.replace('。','')
    line = line.replace('？','')
    line = line.replace('！','')
    line = line.replace('，','')
    line = line.replace('.','')
    line = line.replace(',','')
    line = line.replace('?','')
    line = line.replace('!','')
    line = line.replace('“','')
    line = line.replace('”','')
    line = line.replace('¥', '')
    line = line.replace('@', '')
    line = line.replace('\n', '')
    line = line.replace('(', '')
    line = line.replace(')', '')
    return line


# generate sets of sources and targets from raw data
# sources and targets are basically lists of strings
def generate_dataset(path='./data'):
    dataset1 = open(path + '/xiaohuangji50w_nofenci.conv', encoding="utf-8")

    sentences = []
    for i in dataset1:
        sentences.append(i)

    sources = []
    targets = []

    for i in range(len(sentences)):
        if sentences[i][0] == 'E':
            if '小通' not in sentences[i+1] and '小通' not in sentences[i+2]:
                sources.append(remove_punc(sentences[i+1][2:-1]))
                targets.append(remove_punc(sentences[i+2][2:-1]))

    # dataset2_sources = open(path + '/train.ask.tsv')
    # dataset2_targets = open(path + '/train.answer.tsv')
    #
    # for item in dataset2_sources:
    #     sources.append(remove_punc(item))
    #
    # for item in dataset2_targets:
    #     targets.append(remove_punc(item))

    return sources, targets

def generate_vocab(sources, targets, number_words=20000-4):
    vocab = {}

    for i in range(len(sources)):
        item = sources[i]
        item = remove_punc(item)
        for word in jieba.cut(item):
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for i in range(len(sources)):
        item = targets[i]
        item = remove_punc(item)
        for word in jieba.cut(item):
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # 由于英文不会被分词，所以和数字一起去掉
    vocab_keys = list(vocab.keys())
    for i in range(len(vocab_keys)):
        if vocab_keys[i].encode('UTF-8').isalnum():
            vocab.pop(vocab_keys[i])

    if len(vocab) > number_words:
        sort = sorted(vocab.items(), key=lambda item: item[1])
        num_del = len(vocab) - number_words
        to_del = sort[:num_del]
        for item in to_del:
            vocab.pop(item[0])

    return vocab


# save the customized vocab to a specific path
# add four specifc words to the top of vocab


def save_vocab(vocab, path=''):
    vocab_keys = list(vocab.keys())
    file = open(path+'vocab.txt', 'w', encoding="utf-8")
    file.write('<PAD>' + '\n')
    file.write('<UNK>' + '\n')
    file.write('<S>' + '\n')
    file.write('</S>' + '\n')
    for item in vocab_keys:
        file.write(item + '\n')
    file.close()


# construct index2word and word2index from giving vocab
def load_vocab(vocab_fpath):
    vocab = [line for line in open(vocab_fpath, 'r', encoding="utf-8").read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

# pad the data with <PAD>(0), len(x) should <= length
def pad(data, length, vocab_fpath):
    word2idx, idx2word = load_vocab(vocab_fpath)
    new_data = data
    num_pad = length- len(data)
    for i in range(num_pad):
        new_data.append(word2idx.get('<PAD>'))
    return new_data


# generator function
def generator_fn(sources, targets, vocab_fpath):
    word2idx, idx2word = load_vocab(vocab_fpath)
    for source, target in zip(sources, targets):
        x = []
        y = [2]

        for word in jieba.cut(source):
            x.append(word2idx.get(word, 1))
        x.append(3)
        if len(x) >= hp.maxlen:
            x = x[:20]
            x[-1] = 3
        else:
            x = pad(x, hp.maxlen, vocab_fpath)

        for word in jieba.cut(target):
            y.append(word2idx.get(word, 1))
        y.append(3)
        if len(y) >= hp.maxlen:
            y = y[:20]
            y[-1] = 3
        else:
            y = pad(y, hp.maxlen, vocab_fpath)

        decoder_input, y = y[:-1], y[1:]

        yield (x, len(x), source), (decoder_input, y, len(y), target)

def input_fn(sources, targets, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([None], (), ()),
              ([None], [None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        lambda :generator_fn(sources, targets, vocab_fpath),
        # lambda :generator_fn,
        output_shapes=shapes,
        output_types=types,
        # args=(sources, targets, vocab_fpath)
    )

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever..
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(path, vocab_fpath, batch_size = hp.batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sources, targets = generate_dataset(path)
    batches = input_fn(sources, targets, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sources), batch_size)
    return batches, num_batches, len(sources)

def get_batch_evaluate(query, vocab_fpath):
    targets = ''
    batch = input_fn(query, targets, vocab_fpath, 1, shuffle=False)
    return batch



if __name__ == '__main__':
    generate_dataset()