# import tensorflow as tf
# import numpy as np
# import os
# import time
# import datetime
# import data_helper
# from TextCNN import TextCNN
# from tensorflow.contrib import learn
# import csv
#
#
# # # Parament 参数设置
# # # ========================================
# #
# # # Data Parameters 数据加载参数
# # tf.flags.DEFINE_string("positive_data_file", "./data/rt-polarity.pos", "Data source for the positive data.")
# # tf.flags.DEFINE_string("negative_data_file", "./data/rt-polarity.neg", "Data source for the negative data.")
# #
# #
# # # Eval Parameters 验证参数
# # tf.flags.DEFINE_integer("batch_size",16, "Batch Size (default: 64)")
# # tf.flags.DEFINE_string("checkpoint_dir", "./runs/1516153490/checkpoints", "Checkpoint directory from training run")
# # #指定是否在训练集和测试集上进行验证，反之使用给出的两条数据
# # tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
# #
# #
# # # Misc Parameters设备参数
# # tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# # tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# #
# # FLAGS = tf.flags.FLAGS
# # FLAGS._parse_flags()
# # print('\nParameters:')
# # for attr, value in sorted(FLAGS.__flags.items()):
# #     print('{}={}'.format(attr.upper(),value))
# # print('')
#
# # 改变这里:加载数据。加载自己的数据
# def run(positive_data_file,negative_data_file,if_eval=True,checkpoint_dir='./runs/1548474433/checkpoints',
#         allow_soft_placement=True,log_device_placement=False):
#     if if_eval:
#         x_raw, y_test = data_helper.load_data_and_labels(positive_data_file,negative_data_file)
#         y_test = np.argmax(y_test,axis=1)
#     else:
#         x_raw = ['a masterpiece four years in the making','everying is off']
#         y_test = [1,0]
#
#
#     # map data into vocabulary
#     vocab_path = os.path.join(checkpoint_dir,'..','vocab')
#     print(vocab_path)
#     vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#     x_test = np.array(list(vocab_processor.transform(x_raw)))
#
#     print('\n Evaluating...\n')
#
#     # Evaluation
#     # ====================================
#     # 得到最近保存的模型
#     checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
#     graph = tf.Graph()
#
#     with graph.as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=allow_soft_placement,
#             log_device_placement=log_device_placement
#         )
#
#         sess = tf.Session(config=session_conf)
#         with sess.as_default():
#             saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
#             saver.restore(sess,checkpoint_file)
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
#             batches = data_helper.batch_iter(list(x_test))
#
#             # collect the predictions here
#             all_predictions = []
#
#             for x_test_batch in batches:
#                 batch_predictions = sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.0})
#                 all_predictions = np.concatenate([all_predictions,batch_predictions])
#
#     # print accuracy if y_test is defined
#     if y_test is not None:
#         correct_predictions = float(sum(all_predictions == y_test))
#         print('Total number of test examples: {}'.format(y_test))
#         print('Accuracy: {}'.format(correct_predictions / float(len(y_test))))
#
#     # Save the evalution to a csv
#     predictions_human_readable = np.column_stack((np.array(x_raw),all_predictions))
#     out_path = os.path.join(checkpoint_dir,'..','prediction.csv')
#     print('Saving evaluation to {}'.format(out_path))
#     with open (out_path,'w') as f:
#         csv.writer(f).writerows(predictions_human_readable)


# todo 待解决：tf.train.Saver()的时候无法创建新的saver，但是使用td.train.import_meta_graph

import tensorflow as tf
import numpy as np
import os
import data_helper
from tensorflow.contrib import learn
import csv

# 改变这里:加载数据。加载自己的数据
positive_data_file = './data/rt-polarity.pos'
negative_data_file = './data/rt-polarity.neg'
if_eval = True
checkpoint_dir = './runs/1548567747'
allow_soft_placement=True
log_device_placement=False
batch_size = 16


if if_eval:
    x_raw, y_test = data_helper.load_data_and_labels(positive_data_file,negative_data_file)
    y_test = np.argmax(y_test,axis=1)
else:
    x_raw = ['a masterpiece four years in the making','everying is off']
    y_test = [1,0]


# map data into vocabulary
vocab_path = os.path.join(checkpoint_dir,'vocab')
print(vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print('\n Evaluating...\n')

# Evaluation
# ====================================
# 得到最近保存的模型
graph = tf.Graph()

with tf.device('/gpu:0'):
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement
        )

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir,'checkpoints','save_net.ckpt.meta'))
            saver.restore(sess,os.path.join(checkpoint_dir, 'checkpoints','save_net.ckpt'))

            # get the placeholders from the graph by name
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            # input_y = graph.get_operation_byname('input_y').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

            # tensor we want to evaluate
            predictions = graph.get_operation_by_name('output/predictioins').outputs[0]

            # generate batches for one epoch
            batches = data_helper.batch_iter(list(x_test),batch_size,1,shuffle=False)

            # collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions,{input_x:x_test_batch,dropout_keep_prob:1.0})
                all_predictions = np.concatenate([all_predictions,batch_predictions])

    # print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print('Total number of test examples: {}'.format(y_test))
        print('Accuracy: {}'.format(correct_predictions / float(len(y_test))))

    # Save the evalution to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw),all_predictions))
    out_path = os.path.join(checkpoint_dir,'..','prediction.csv')
    print('Saving evaluation to {}'.format(out_path))
    with open (out_path,'w') as f:
        csv.writer(f).writerows(predictions_human_readable)