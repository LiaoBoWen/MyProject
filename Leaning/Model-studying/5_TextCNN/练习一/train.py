import numpy as np
import os
import time
import datetime
from multiprocessing import cpu_count
import warnings
import pickle

warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')

import data_helper
from cnn_model import TextCNN

import tensorflow as tf
from tensorflow.contrib import learn

from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

dev_sample_percentage = .2
positive_data_file = '../data/rt-polarity.pos'
negative_data_file = '../data/rt-polarity.neg'
# 模型超参
embedding_dim = 300
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
filter_sizes = [3,4,5]
num_filters = 128
learn_rate = 1e-3
# 数据的部分的参数
batch_size = 32
num_epochs = 10
# 保存部分的参数
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
W2V_path = './w2v_model.pkl'
x_train_w2v = './x_train_w2v.pkl'
x_test_w2v_ = './x_test_w2v.pkl'
vocab_path = './vocab.pkl'
# 启动方面的参数
allow_soft_placement = True
log_device_placement = False
# 随机种子
seed=32


def data_process():
    # 数据的加载
    x_text, y = data_helper.load_data_and_labels('../data/rt-polarity.pos','../data/rt-polarity.neg')
    x_text = np.array(x_text)

    # 打乱数据
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(len(x_text))
    x_text = x_text[shuffle_indices]
    y = y[shuffle_indices]

    # 切分数据
    x_train, x_test, y_train, y_test = train_test_split(x_text,y,test_size=dev_sample_percentage)
    max_document_length = max([len(x.split()) for x in x_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    x_train = vocab_processor.fit_transform(x_train)
    x_test = vocab_processor.transform(x_test)
    vocab_processor.save(vocab_path)


    # 输入word2vec 词表
    x_train = np.array(list(x_train)).astype(np.str)        # 由于word2vec的词表建立需要的是字符串形式，所以这里需要进行转换
    x_test = np.array(list(x_test)).astype(np.str)
    # x_train = [sent.split() for sent in x_train]
    # x_test = [sent.split() for sent in x_test]
    x_train = [list(x) for x in list(x_train)]

    # Word2Vec 模型
    if not os.path.exists(W2V_path):
        print('开始训练词向量...')
        w2v = Word2Vec(size=embedding_dim,window=5,min_count=0,workers=cpu_count())
        w2v.build_vocab(x_train)   # todo 这里（还有min_count）不合理，但是怎么处理未登录词的问题
        w2v.train(x_train,total_examples=w2v.corpus_count,epochs=100)
        w2v.save(W2V_path)
    else:
        print('加载Word2Vec模型...')
        w2v = Word2Vec.load(W2V_path)

    if not os.path.exists(x_train_w2v):
        print('train_word2vec...')
        x_train = [[w2v[w] for w in s ] for s in x_train]
        pickle.dump(x_train,open(x_train_w2v,'wb'))
    else:
        print('加载x_train_w2v...')
        x_train = pickle.load(open(x_train_w2v,'rb'))

    if not os.path.exists(x_test_w2v_):
        print('test_word2vec...')
        x_test = ([([w2v[w] for w in s ]) for s in x_test])
        pickle.dump(x_test,open(x_test_w2v_,'wb'))
    else:
        print('加载x_test_w2v...')
        x_test = pickle.load(open(x_test_w2v_,'rb'))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    print(x_test)

    return x_train, x_test, y_train, y_test


def train_step(sess,cnn,x_batch,y_batch,train_op,step):
    feed_dict = {
        cnn.input_x:x_batch,
        cnn.input_y:y_batch,
        cnn.dropout_keep_prob:dropout_keep_prob
    }

    _, loss, accuracy = sess.run(
        [train_op,cnn.loss,cnn.accuracy],
        feed_dict
    )

    time_str = datetime.datetime.now().isoformat()
    print('{}:step {}, loss {}, acc: {}'.format(time_str,step,loss,accuracy))
    return accuracy

def dev_step(sess,cnn,x_batch,y_batch,step):
    feed_dict = {
        cnn.input_x:x_batch,
        cnn.input_y:y_batch,
        cnn.dropout_keep_prob:1.
    }
    loss, accuracy  = sess.run(
        [cnn.loss,cnn.accuracy],
        feed_dict
    )
    time_str = datetime.datetime.now().isoformat()
    print('{}:step {}, loss {}, acc: {}'.format(time_str,step,loss,accuracy))
    return accuracy

def train():
    x_train, x_test, y_train, y_test = data_process()
    print('x_train`s shape:',x_train.shape)
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement = allow_soft_placement,
                log_device_placement = log_device_placement
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length = x_train.shape[1],
                    num_classes = y_train.shape[1],
                    filter_sizes = filter_sizes,
                    num_filters = num_filters,
                    l2_reg_lambda = l2_reg_lambda,
                    embedding_size=embedding_dim)

                global_step = 0

                train_op = tf.train.AdamOptimizer(learn_rate).minimize(cnn.loss)

                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir,timestamp))
                print('Writing to {}'.format(out_dir))

                checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver()

                # initialize all variables
                # 全局初始化
                sess.run(tf.global_variables_initializer())

                batches = data_helper.batch_iter(
                    list(zip(x_train,y_train)),
                    batch_size,
                    num_epochs
                )

                best_acc = 0
                for batch in batches:
                    global_step += 1
                    x_batch, y_batch = zip(*batch)
                    train_step(sess,cnn,x_batch,y_batch,train_op,global_step)
                    if global_step % evaluate_every == 0:
                        print('\n Evaluation:')
                        accurent_acc = dev_step(sess,cnn,x_test,y_test,global_step)
                        print('')
                    if global_step % checkpoint_every == 0 and accurent_acc > best_acc:
                        path = saver.save(sess,checkpoint_dir + '/save_net.ckpt')
                        best_acc = accurent_acc
                        print('Saved model checkpoint to {}\n'.format(path))


if __name__ == '__main__':
    train()