import jieba
import pickle
import numpy as np
import os
from hyperparams import Params


def clear_punc(line):
    line = line.replace(',','')
    line = line.replace('.','')
    line = line.replace('，','')
    line = line.replace('。','')
    line = line.replace('？','')
    line = line.replace('！','')
    line = line.replace('\n','')
    line = line.replace('‘','')
    line = line.replace('’','')
    line = line.replace('“','')
    line = line.replace('”','')
    line = line.replace('"','')
    line = line.replace("'",'')
    line = line.replace("#",'')
    line = line.replace("@",'')

    return line


def get_data(path='../data/xiaohuangji50w_nofenci.conv'):
    sentences = []
    sources = []
    targets = []

    with open(path,encoding='utf8') as f:

        for line in f.readlines():
            sentences.append(line)

        for i in range(len(sentences)):
            if sentences[i][0] == 'E':
                sentences_1 = sentences[i+1]
                sentences_2 = sentences[i+2]
                if '小通' not in sentences_1 and '小通'  not in sentences_2:
                    sources.append(clear_punc(sentences_1[2:-1].strip()))
                    targets.append(clear_punc(sentences_2[2:-1].strip()))

    return sources, targets


def make_vocab(sources, targets, vocab_size=32288 - 4):
    vocab = {}

    for source in sources:
        for word in jieba.cut(source):
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for target in targets:
        for word in jieba.cut(target):
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1


    if len(vocab) > vocab_size:
        items = sorted(vocab.items(),key=lambda x:x[1])
        del_num = len(vocab) - vocab_size
        for item in items[:del_num]:
            vocab.pop(item[0])

    return vocab


def save_vocab(vocab,idx2token_path='./model/idx2token.pkl',token2idx_path='./model/token2idx.pkl'):
    token2idx = {val:idx for idx, val in enumerate(vocab,start=4)}
    idx2toekn = {idx:val for idx, val in enumerate(vocab,start=4)}
    token2idx['<PAD>'] = 0
    token2idx['<UNK>'] = 1
    token2idx['<S>'] = 2
    token2idx['</S>'] = 3

    idx2toekn[0] = '<PAD>'
    idx2toekn[1] = '<UNK>'
    idx2toekn[2] = '<S>'
    idx2toekn[3] = '</S>'

    if not os.path.exists('./model'):
        os.mkdir('./model')
    pickle.dump(idx2toekn,open(idx2token_path,'wb'))
    pickle.dump(token2idx,open(token2idx_path,'wb'))

    print('Save vocab to path:./model/ successfuly .')


def load_vocab(id2token_path='./model/idx2token.pkl',token2id_path='./model/token2idx.pkl'):
    id2token = pickle.load(open(id2token_path,'rb'))
    token2id = pickle.load(open(token2id_path,'rb'))

    print('Load vocab from path:./model/ successfuly .')
    return id2token, token2id


def save_data_to_pickle(data,data_path='./data/data'):
    if not os.path.exists('./data'):
        os.mkdir('./data')

    pickle.dump(data,open(data_path,'wb'))

    print('Save processed data to path: ./data/data')


def load_processed_data(data_path='./data/data'):
    with open(data_path,'rb') as f:
        data = pickle.load(f)

    print('Load data successfuly from path:./data/data .')
    return data


def make_token(sources,targets,id2token_path='./model/idx2token.pkl',token2id_path='./model/token2idx.pkl'):

    id2token, token2id = load_vocab(id2token_path,token2id_path)
    def pad(x,max_len,pad_token=0):
        x = x + [pad_token] * (max_len - len(x))
        return x

    xs = []

    decode_inputs = []
    ys = []

    for source, target in zip(sources,targets):
        x = []
        y = [2]

        for word in jieba.cut(source):
            x.append(token2id.get(word,1))
        x.append(3)

        if len(x) > Params.maxlen:
            x = x[:20]
            x[-1] = 3
        else:
            x = pad(x,Params.maxlen)

        for word in jieba.cut(target):
            y.append(token2id.get(word,1))
        y.append(3)

        if len(y) > Params.maxlen:
            y = y[:20]
            y[-1] = 3
        else:
            y = pad(y,Params.maxlen)

        decode_input, y = y[:-1], y[1:]

        xs.append(x)

        decode_inputs.append(decode_input)
        ys.append(y)

    print('Data token finished .')
    return (xs, decode_inputs, ys)


def get_batch(data,epoch=50,batch_size=32,shuffle=True):
    xs, decode_inputs, ys = data

    xs = np.array(xs)
    decode_inputs = np.array(decode_inputs)
    ys = np.array(ys)

    batch_num = (xs.shape[0] - 1) // batch_size + 1
    for epoch_i in range(epoch):
        if shuffle:
            shuffle_idx = np.random.permutation(xs.shape[0])
            xs = xs[shuffle_idx]
            decode_inputs = decode_inputs[shuffle_idx]
            ys = ys[shuffle_idx]

        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, xs.shape[0])
            yield xs[start:end], decode_inputs[start:end], ys[start:end]



if __name__ == '__main__':
    # sources, targets = get_data()
    # vocab = make_vocab(sources,targets)
    # save_vocab(vocab)

    # data = make_token(sources,targets)
    # save_data_to_pickle(data)

    # todo Note:data = [x, len(x), decode_input, y, len(y)] decode_input start with <S> and y end with </S> , in a say: y is behind in decode_input
    data = load_processed_data()

    for xs, decode_inputs, ys in get_batch(data,2,4,shuffle=False):
        print('xs:')
        print(xs)
        print('decode_inputs:')
        print(decode_inputs)
        print('ys:')
        print(ys)
        break