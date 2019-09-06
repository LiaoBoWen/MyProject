 # todo 待解决：tf.train.Saver()的时候无法创建新的saver，但是使用td.train.import_meta_graph
import csv
import os
import warnings
import numpy as np
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
warnings.filterwarnings(action='ignore',category=FutureWarning,module='gensim')

import data_helper

from gensim.models.word2vec import Word2Vec

import tensorflow as tf
from tensorflow.contrib import learn

# 改变这里:加载数据
positive_data_file = '../data/rt-polarity.pos'
negative_data_file = '../data/rt-polarity.neg'
if_eval = True
checkpoint_dir = './1551590814'
allow_soft_placement=True
log_device_placement=False
batch_size = 32


if if_eval:
    x_raw, y_test = data_helper.load_data_and_labels(positive_data_file,negative_data_file)
    y_test = np.argmax(y_test,axis=1)
    print(y_test)
else:
    x_raw = ['a masterpiece four years in the making','everying is off']
    y_test = [1,0]


# map data into vocabulary
vocab_processor = learn.preprocessing.VocabularyProcessor.restore('./vocab.pkl')
w2v = Word2Vec.load('./w2v_model.pkl')

x_test = np.array(list(vocab_processor.transform(x_raw))).astype(np.str)    # 由于word2vec的词表建立需要的是字符串形式，所以这里需要进行转换
x_test = [list(x) for x in list(x_test)]

x_test = [[w2v[w] for w in s] for s in x_test]


print('\n Evaluating...\n')

# Evaluation
# ====================================
# 得到最近保存的模型
# graph = tf.Graph()
# with tf.device('/gpu:0'):
#     with graph.as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=allow_soft_placement,
#             log_device_placement=log_device_placement
#         )
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir,'checkpoints','save_net.ckpt.meta'))
#             saver.restore(sess,os.path.join(checkpoint_dir, 'checkpoints','save_net.ckpt'))
#
#             # get the placeholders from the graph by name
#             input_x = graph.get_operation_by_name('input_x').outputs[0]
#             # input_y = graph.get_operation_byname('input_y').outputs[0]
#             dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
#
#             # tensor we want to evaluate
#             predictions = graph.get_operation_by_name('output/predictions').outputs[0]
#
#             # generate batches for one epoch
#             batches = data_helper.batch_iter(list(x_test),batch_size,1,shuffle=False)
#
#             # collect the predictions here
#             all_predictions = []
#
#             for x_test_batch in batches:
#                 batch_predictions = sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.0})
#                 all_predictions = np.concatenate([all_predictions,batch_predictions])


    # # print accuracy if y_test is defined
    # if y_test is not None:
    #     correct_predictions = float(sum(all_predictions == y_test))
    #     print('Total number of test examples: {}'.format(y_test))
    #     print('Accuracy: {}'.format(correct_predictions / float(len(y_test))))
    #
    # # Save the evalution to a csv
    # predictions_human_readable = np.column_stack((np.array(x_raw),all_predictions))
    # out_path = os.path.join(checkpoint_dir,'..','prediction.csv')
    # print('Saving evaluation to {}'.format(out_path))
    # with open (out_path,'w') as f:
    #     csv.writer(f).writerows(predictions_human_readable)


    # Accuracy: 0.8335834896810507





graph = tf.Graph()
with tf.device('/gpu:0'):
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, 'checkpoints', 'save_net.ckpt.meta'))
            saver.restore(sess, os.path.join(checkpoint_dir, 'checkpoints', 'save_net.ckpt'))

            input_x = graph.get_operation_by_name('input_x').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            predictions = graph.get_operation_by_name('output/predictions').outputs[0]
            batches = data_helper.batch_iter(x_test,batch_size,1,shuffle=False)

            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.})
                all_predictions = np.concatenate([all_predictions,batch_predictions])

    if y_test is not None:
        correct_predictions = float(sum(y_test == all_predictions))
        print('Total number of test examples: {}'.format(y_test.shape[0]))
        print('Accuracy: {}'.format(correct_predictions / float(y_test.shape[0])))

    predictions_human_readable = np.column_stack((np.array(x_raw),all_predictions))
    out_path = os.path.join(checkpoint_dir,'..','prediction.csv')
    print('Saving evaluation to {}'.format(out_path))
    with open(out_path,'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
