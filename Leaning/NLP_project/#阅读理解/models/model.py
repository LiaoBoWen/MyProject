import tensorflow as tf
import numpy as np
import pickle as pkl


from models import SiameseNN
from utils import *
import os

class NNConfig:
    def __init__i(self,vocab_size,embeddings=None):
        # input_length
        self.max_q_len = 200
        self.max_a_len = 200
        # other size
        self.num_epochs = 100
        self.batch_size = 128
        # vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = 256
        self.output_size = 128
        self.keep_prob = 0.6

        self.embeddings = embeddings
        self.embedding_size = 100
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]
        self.lr = 1e-3
        self.pos_weight = 0.25

        self.cf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction=0.2


def train(train_corpus,config,val_corpus,eval_train_corpus=None):
    iterator = Iterator(train_corpus)
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)
    with tf.Session() as sess:
        model = SiameseNN(config)
        saver = tf.train.Saver(max_to_keep=3)
        sess.run(tf.global_variables_initializer())
        for epoch in range(config.num_epochs):
            count = 0
            for batch_x in iterator.next(config.batch_size,shuffle=True):
                batch_qid, batch_q, batch_aids, batch_a, labels = zip(*batch_x)
                batch_q = np.array(batch_q)
                batch_a = np.array(batch_a)
                _, loss = sess.run([model.train,model.total_loss],
                                   feed_dict={
                                       model.q: batch_q,
                                       model.a:batch_a,
                                       model.labels:labels,
                                       model.keep_prob:config.keep_prob,

                                   })

                if count % 50:
                    print('epoch:{} batch_size:{} loss:{}'.format(epoch,count,loss))
            saver.save(sess,config.model_path,global_step=epoch)



def evaluate(sess,model,val_corpus,config):
    iterator = Iterator(val_corpus)

    count = 0

    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = []

    total_loss = 0
    for batch_x in iterator.next(config.batch_size,shuffle=False):
        batch_qids, batch_q, batch_aids, batch_a, labels = zip(*batch_x)

        batch_q = np.array(batch_q)
        batch_a = np.array(batch_a)


        q_a_cosine, loss = sess.run([model.q_a_cosine,model.loss],
                                    feed_dict={
                                        model.q: batch_q,
                                        model.a: batch_a,
                                        model.y: labels,
                                        model.keep_prob: config.keep_prob
                                    })
        total_loss += loss
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(q_a_cosine)
        total_labels.append(labels)

    total_qids = np.concatenate(total_qids)
    total_aids = np.concatenate(total_aids)
    total_pred = np.concatenate(total_pred)
    total_labels= np.concatenate(total_labels)

    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    return 'MAR:{} ,MRR:{}'.format(MAP,MRR)

def test(corpus,config):
    with tf.Session(config.cf) as sess:
        model = SiameseNN()
        saver = tf.train.Saver()
        latest_model = tf.train.latest_checkpoint(config.model_path)
        saver.restore(sess,latest_model)
        print('[test]',evaluate(sess,model,corpus,config))

def main(args):
    max_q_len = 25
    max_a_len = 90
    with open(os.path.join(processed_data_path,'pointwise_corpus.pkl'),'r') as f:
        train_corpus, val_corpus, test_corpus = pkl.load(f)
        embeddings = build_embedding(embedding_path,word2id)
        train_qids, train_q, train_aids, train_a, train_labels = zip(*train_corpus)
        train_q = padding(train_q,max_q_len)
        train_a = padding(train_a,max_a_len)
        train_corpus = zip(train_qids,train_q,train_aids,train_a,train_labels)

        val_qids, val_q, val_aids, val_a, val_labels = zip(*val_corpus)
        val_q = padding(val_q,max_q_len)
        val_a = padding(val_a,max_a_len)
        val_corpus = zip(val_qids, val_q, val_aids, val_a, val_labels)

        test_qids, test_q, test_aids, test_a, labels = zip(*test_corpus)
        test_q = padding(test_q)
        test_a = padding(test_a)
        test_corpus = zip(test_qids,test_q,test_aids,test_a,labels)

        config = NNConfig(max(word2id.values()) + 1,embeddings=embeddings)
        config.max_q_len = max_q_len
        config.max_a_len = max_a_len

        if args.train:
            train(deepcopy(train_corpus),config,val_corpus,deepcopy(train_corpus))
        elif args.test:
            test(test_corpus,config)

if __name__ == '__main__':
    class args:
        train = True
        test = False

    model_path = 'models'
    raw_data_path = '../data/WikiQA/raw'
    processed_data_path = '../data/WikiQA/processed'
    embedding_path = '../data/embedding/glove.6B.300d.txt'

    with open(os.path.join(processed_data_path),'r') as f:
        word2id, id2word = pkl.load(f)
    main(args)

