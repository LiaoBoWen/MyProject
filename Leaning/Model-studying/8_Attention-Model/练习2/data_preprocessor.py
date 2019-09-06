import numpy as np

def preprocess_data(dataset,human_vocab,machine_vocab,Tx,Ty):
    # Metadata
    m = len(dataset)

    # Initialize
    X = np.zeros([m,Tx],dtype='int32')
    Y = np.zeros([m,Ty],dtype='int32')

    # Process data char2id

    for i in range(m):
        data = dataset[i]
        X[i] = np.array(tokenize(data[0],human_vocab,Tx))
        Y[i] = np.array(tokenize(data[1],machine_vocab,Ty))

    Xoh = oh_2d(X,len(human_vocab))
    Yoh = oh_2d(Y,len(machine_vocab))

    return X, Y, Xoh, Yoh

def tokenize(sentence,vocab,length):
    tokens = [0] * length
    for i in range(length):
        char = sentence[i] if i < len(sentence) else '<pad>'
        char = char if (char in vocab) else '<unk>'
        tokens[i] = vocab[char]

    return tokens

def ids_to_keys(sequence, vocab):
    return [list(vocab.keys())[id] for id in sequence]

def oh_2d(dense,max_value):
    oh = np.zeros(np.append(dense.shape,[max_value]))
    ids1, ids2 = np.meshgrid(np.arange(dense.shape[0]),np.arange(dense.shape[1]))

    oh[ids1.flatten(),ids2.flatten(),dense.flatten('F').astype(int)] = 1

    return oh