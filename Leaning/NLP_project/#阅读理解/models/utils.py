import numpy as np
import tensorflow as tf

def padding(data,max_len):
    return tf.keras.preprocessing.sequence.pad_sequences(data,max_len,padding='post',truncating='post')

def eval_map_mrr(qids,aids,preds,labels):
    '''what'''
    dic = dict()
    pred_dict = dict()
    for qid, aid, pred, label in zip(qids, aids, preds, labels):
        pred_dict.setdefault(qid,[])
        pred[qid].append([aid,pred,label])

    for qid in pred_dict:
        dic[qid] = sorted(pred_dict[qid],key=lambda x:x[1],reverse=True)
        aid2rank = {aid:[label,rank] for (rank, (aid, pred, label)) in enumerate(dic[qid])}
        dic[qid] = aid2rank

    MAP, MRR = .0 ,.0
    useful_q_len = 0
    for q_id in dic:
        sort_rank = sorted(dict[q_id].items(),key=lambda x:x[1][1],reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            if sort_rank[i][1][0] == 1:
                correct += 1
        if correct == 0:
            continue
        useful_q_len += 1
        correct = 0
        for i in range(len(sort_rank)):
            if sort_rank[i][1][0] == 1 and mrr_mark == False:
                MRR += 1. / float(i + 1)
                mrr_mark = True

            total += 1
            if sort_rank[i][1][0] == 1:
                correct +=1
                AP += float(correct) / float(total)

        MAP /= useful_q_len
        MRR /= useful_q_len
        return MAP, MRR

def build_embedding(in_file,word_dict):
    num_words = max(word_dict.values()) + 1
    dim = int(in_file.split('.')[-2][:-1])
    embeddings = np.zeros((num_words,dim))

    if in_file is not None:
        pre_trained = 0
        initialized = {}
        avg_sigma = 0
        avg_mu = 0

        for line in open(in_file).readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                initialized[sp[0]] = True
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float()]
                mu = embeddings[word_dict[sp[0]]].mean()
                sigma = np.std(embeddings[word_dict[sp[0]]])
                avg_mu = mu
                avg_sigma += sigma
        avg_sigma /= 1. * pre_trained
        avg_mu /= 1. * pre_trained
        for w in word_dict:
            if w not in initialized:
                embeddings[word_dict[w]] = np.random.normal(avg_mu,avg_sigma,(dim,))
        print('Pre-trained:{} {:.2f}'.format(pre_trained,pre_trained * 100. / num_words))

    return embeddings.astype(np.float32)

class Iterator:
    def __init__(self,x):
        self.x = x
        self.sample_num = len(self.x)

    def num_batch(self,batch_size,shuffle=True):
        if shuffle:
            np.random.shuffle(self.x)

        l = np.random.randint(0,self.sample_num - batch_size + 1)
        r = l + batch_size
        x_part = self.x[l:r]

        return x_part

    def next(self,batch_size,shuffle=False):
        if shuffle:
            np.random.shuffle(self.x)
        l = 0
        while l < self.sample_num:
            r = min(l + batch_size,self.sample_num)
            batch_size = r - l
            x_part = self.x[l:r]
            l += batch_size
            yield x_part
