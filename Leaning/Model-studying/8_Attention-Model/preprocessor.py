import numpy as np


def preprocess_data(dataset,human_vocab,machine_vocab,Tx,Ty):
    '''
    :param dataset:a list of sentence data pairs
    :param human_vocab: a dictionary of tokens(chars) to id's
    :param machine_vocab: a dictionary of tokens(chars) to id's
    :param Tx: x data size
    :param Ty: y data size
    :return: X: Sparse tokens for x data; Y: sparse tokens for Y data; Xoh: one-hot tokens for x data; Yoh: one-hot tokkens for y data
    '''
    # Metadata
    m = len(dataset)

    # Initialize
    X = np.zeros([m,Tx],dtype='int32')
    Y = np.zeros([m,Ty],dtype='int32')

    # Process data  char2id
    for i in range(m):
        data = dataset[i]
        X[i] = np.array(tokenize(data[0],human_vocab,Tx))
        Y[i] = np.array(tokenize(data[1],machine_vocab,Ty))

    # Expand ont-hot
    Xoh = oh_2d(X,len(human_vocab))     # 大费周章的就是为了把id进行one-hot化
    Yoh = oh_2d(Y,len(machine_vocab))

    return (X,Y,Xoh,Yoh)


def tokenize(sentence,vocab, length):
    '''
    :param sentence: series of tokens
    :param vocab: a dictionary from token to id
    :param length: max number of tokens to consider
    :return: a series fo id's for given input token sequence
    '''
    tokens = [0] * length
    for i in range(length):
        char = sentence[i] if i < len(sentence) else "<pad>"        # 基于字符的预测
        char = char if (char in vocab) else "<unk>"
        tokens[i] = vocab[char]

    return tokens


def ids_to_keys(sequence,vocab):
    '''
    converts a series of id's into the keys of a dictionary
    :param sequence:
    :param vocab:
    :return:
    '''
    return [list(vocab.keys())[id] for id in sequence]


def oh_2d(dense,max_value):
    # create a one-hot array for the 2d input dense array

    # Initialize
    oh = np.zeros(np.append(dense.shape,[max_value]))

    # Set correct indices
    ids1, ids2 = np.meshgrid(np.arange(dense.shape[0]),np.arange(dense.shape[1]))   # todo np.meshgrid ?

    '''
    np.meshgrid(np.arange(3),np.arange(4))[1]
Out[37]: 
array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])
       np.meshgrid(np.arange(3),np.arange(4))[0]
Out[38]: 
array([[0, 1, 2],
       [0, 1, 2],
    '''

    oh[ids1.flatten(),ids2.flatten(),dense.flatten('F').astype(int)] =1

    return oh