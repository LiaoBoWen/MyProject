import numpy as np
import pickle

tag2label = {'O':0,
             'B-ORG':1,'I-ORG':2,
             'B-LOC':3,'I-LOC':4,
             'B-PER':5,'I-PER':6}



def read_corpus(data_path):
    data = []

    sent = []
    tags = []

    with open(data_path,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                word, tag = line.strip().split()
                sent.append(word)
                tags.append(tag2label[tag])
            else:
                data.append([sent,tags])
                sent, tags = [] , []

    return data


def build_word2id(dict_path,data_path,min_freq):
    data = read_corpus(data_path)

    word2id = {}
    word2id_len = 0

    for sent, label in data:
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            elif 'a' <= word <= 'z' or 'A' <= word <= 'Z':
                word = '<ENG>'
            # 同时记录出现次数以及id
            if word not in word2id:
                word2id_len += 1
                word2id[word] = [word2id_len, 1]
            else:
                word2id[word][1] += 1

    # 去除低频率的词（其实就是字，因为序列标注是基于字级别）
    low_freq_words = []
    for word, [word_id, freq] in word2id.items():
        if freq < min_freq and word != '<NUM>' and word != 'NEG':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id =1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1

    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0


    id2word = dict(zip(word2id.values(),word2id.keys()))

    with open(dict_path,'wb') as f:
        pickle.dump(word2id,f)
    with open('./data/id2word.pkl','wb') as f:
        pickle.dump(id2word,f)


def get_word2id(word2id_path='./data/word2id.pkl',id2word_path='./data/id2word.pkl'):
    with open(word2id_path,'rb') as f:
        word2id = pickle.load(f)
    with open(id2word_path,'rb') as f:
        id2word = pickle.load(f)

    return word2id,id2word

def sentence2id(sentence,word2id):
    sentence_id = []
    for word in sentence:
        if word.isdigit():
            word = '<NUM>'
        elif 'a' <= word <= 'z' and 'A' <= word <= 'Z':
            word = '<ENG>'

        sentence_id.append(word2id.get(word,word2id['<UNK>']))

    return sentence_id

def pad_sentences(sentence_batch,pad_mark=0):
    max_len = max(map(lambda sentence:len(sentence),sentence_batch))

    sent, len_sent = [], []
    for sentence in sentence_batch:
        len_ = len(sentence)
        seq = sentence + [pad_mark] * (max_len - len_)
        sent.append(seq)
        len_sent.append(len_)

    return sent, len_sent


def generate_batch(data,word2id,epoch_num,batch_size,shuffle=False):
    length = len(data)

    for epoch_ in range(epoch_num):
        if shuffle:
            np.random.shuffle(data)

        batch_num = ( length - 1 ) // batch_size + 1

        for i in range(batch_num):
            sents_batch, labels_batch = [], []

            start = i * batch_size
            end = min(i * batch_size + batch_size, length)

            for sent, labels in data[start:end]:
                sents_batch.append(sentence2id(sent,word2id))
                labels_batch.append(labels)

            sents_batch, sents_lens = pad_sentences(sents_batch)
            labels_batch, _ = pad_sentences(labels_batch)
            # sents_batch = np.array(sents_batch)
            # labels_batch = np.array(labels_batch)
            yield sents_batch, labels_batch, sents_lens, epoch_




if __name__ == '__main__':
    '''test'''
    data = read_corpus('./data/test_data')
    build_word2id('./data/word2id.pkl','./data/train_data',3)
    word2id,id2word = get_word2id('./data/word2id.pkl')
    print(len(word2id))
    print(id2word)
    # for sents_batch, labels_batch, sents_lens in generate_batch(data,word2id,20,64):
    #     print(sents_batch, labels_batch, sents_lens)
    #     break