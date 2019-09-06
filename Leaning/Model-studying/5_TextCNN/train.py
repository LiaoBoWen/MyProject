# import  tensorflow as tf
# import numpy as np
# import os
# import time
# import datetime
# import data_helper
# from TextCNN import TextCNN
# from tensorflow.contrib import learn # todo learn.preprocessing.VocablaryProcessor()
#
# todo tf.flags.DEFINE_xxx(),用于命令行参数的传递
# todo 存在一个问题：没有模型读取的部分，当中断程序后重新运行需要从头训练
#
# # # Parameters 参数设置
# # # =============================================
# #
# # # Data loading params.json 数据加载参数
# # tf.flags.DEFINE_float('dev_sample_percentage',.1,'Percentage of the training data to use for validation')
# # tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# # tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# # tf.flags.DEFINE_string('positive_data_file','./data/rt-polarity.pos','Data source for the positive data')
# # tf.flags.DEFINE_string('negative_data_file','./data/rt-polarity.neg','Data source for the negative data')
# #
# # # Model Hyperparameters 超参数设置
# #
# # tf.flags.DEFINE_integer('embedding_dim',128,'Dimensionality of character embedding (default:128)')
# # tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# # tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# #
# # # Training parameters 训练参数
# # tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
# # tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
# # tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# # #每一百轮便保存模型
# # tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# # #仅保存最近五次模型
# # tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# # # Misc Parameters 设备参数
# # #当指定的设备不存在时，自动分配（默认为TRUE）
# # tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# # #打印日志
# # tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# #
# # FLAGS = tf.flags.FLAGS
# # FLAGS._parse_args_flags()
# #
# # print('Parameters:')
# # for attr, value in sorted(FLAGS.__flags.items()):
# #     print('{} ={}'.format(attr.upper(),value))
# # print('')
#
# dev_sample_percentage = .1
# positive_data_file = './data/rt-polarity.pos'
# negative_data_file = './data/rt-polarity.neg'
# embedding_dim = 128
# dropout_keep_prob = 0.5
# l2_reg_lambda = 0.0
# batch_size = 16
# num_epochs = 10
# evaluate_every = 100
# checkpoint_every = 100
# num_checkpoints = 5
# allow_soft_placement = True
# log_device_placement = False
# filter_sizes = '3,4,5'
# num_filters = 128
#
# # Data Preparation
# # ==============================================
#
# # Load data加载数据，返回数据集和标签
# print('Loading data...')
# x_text, y = data_helper.load_data_and_labels(positive_data_file,negative_data_file)
#
# # Build vocabulary 生成但是字典
# # 得到最大邮件长度（单词个数），不足的用0补充
# max_document_length = max([len(x.split(' ')) for x in x_text])
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x = np.array(list(vocab_processor.fit_transform(x_text)))       # todo 生成word_to_id-Metrix ，不够的补零
#
# # 数据打乱数据集
# np.random.seed(32)
#
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffle = x[shuffle_indices]
# y_shuffle = y[shuffle_indices]
#
# # train_test_split  处理交粗糙，应当使用cross-validaton
# # 从后往前取
# dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
# y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]
#
# print('Vocabulary Size:{}'.format(len(vocab_processor.vocabulary_))) # todo 计算所有的词汇数量
# print('Train/Dev split:{}/{}'.format(len(y_train),len(y_dev)))
#
#
# def train_step(x_batch,y_batch,train_op):
#     '''
#     a single training step
#     '''
#     feed_dict = {
#         cnn.input_x: x_batch,
#         cnn.input_y: y_batch,
#         cnn.dropout_keep_prob: dropout_keep_prob
#     }
#
#     _, step, summaries, loss, accuracy = sess.run(
#         [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
#     feed_dict)
#
#     time_str = datetime.datetime.now().isoformat()
#     print('{}:step {}, loss {}, acc: {}'.format(time_str,step,loss,accuracy))
#     train_summary_writer.add_summary(summaries,step)
#
# def dev_step(x_batch, y_batch,writer=None):
#     '''
#     evalutes model on dev set
#     '''
#     feed_dict = {
#         cnn.input_x: x_batch,
#         cnn.input_y: y_batch,
#         cnn.dropout_keep_prob: 1.0
#     }
#
#     step, summaries, loss, accuracy = sess.run(
#         [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
#         feed_dict)
#
#     time_str = datetime.datetime.now().isoformat()
#     print('{}: step {}, loss {}, acc {}'.format(time_str,step,loss,accuracy))
#
#     if writer:
#         writer.add_summary(summaries,step)
# with tf.device('/gpu:0'):
#     with tf.Graph().as_default():
#         # session
#         session_conf = tf.ConfigProto(
#             allow_soft_placement = allow_soft_placement,
#             log_device_placement = log_device_placement)
#         sess = tf.Session(config=session_conf)
#
#         with sess.as_default():
#             cnn = TextCNN(
#                 sequence_length = x_train.shape[1], # 句子的长度。记住，已经填补了所有的句子的长度（我们的数据集为59）
#                 num_classes = y_train.shape[1],
#                 vocab_size = len(vocab_processor.vocabulary_),   # 邮件字典不重复单词数目
#                 embedding_size = embedding_dim,   # 中间层向量，也就是词嵌入的维度
#                 filter_sizes = list(map(int,filter_sizes.split(','))), # 卷积核大小
#                 num_filters = num_filters,    # 卷积核数目
#                 l2_reg_lambda = l2_reg_lambda) # l2惩罚项大小
#
#             global_step = tf.Variable(0,name='global_step',trainable=False)
#
#             # train_op = tyf.train.AdamOptimizer(1e-3).minimize(cnn.loss)
#             optimizer = tf.train.AdamOptimizer(1e-3)
#             grads_and_vars = optimizer.compute_gradients(cnn.loss)
#             train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
#
#             # keep track of gradient values and sparsity (optional)
#             grad_summaries = []
#             for g, v in grads_and_vars:
#                 if g is not None:
#                     grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
#                     sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
#                     grad_summaries.append(grad_hist_summary)
#                     grad_summaries.append(sparsity_summary)
#             grad_summaries_merged = tf.summary.merge(grad_summaries)
#
#             # Output directory for models and summaries
#             timestamp = str(int(time.time()))
#             out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
#             print("Writing to {}\n".format(out_dir))
#
#             # Summaries for loss and accuracy
#             loss_summary = tf.summary.scalar("loss", cnn.loss)
#             acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
#
#             # Train Summaries
#             train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
#             train_summary_dir = os.path.join(out_dir, "summaries", "train")
#             train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
#
#             # Dev summaries
#             dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
#             dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
#             dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
#
#             # checkpoint dirctory . tensorflow assumes the directory already exists so we need to create it
#             checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))
#             checkpoint_prefix = os.path.join(checkpoint_dir,'model')
#             if not os.path.exists(checkpoint_dir):
#                 os.makedirs(checkpoint_dir)
#             saver = tf.train.Saver(tf.global_variables(),max_to_keep=num_checkpoints)
#
#             # write vocabulary
#             vocab_processor.save(os.path.join(out_dir,'vocab'))
#
#             # initialize all variables
#             # 全局初始化
#             sess.run(tf.global_variables_initializer())
#
#             # generate batch
#             batches = data_helper.batch_iter(
#                 list(zip(x_train,y_train)),
#                 batch_size,
#                 num_epochs
#             )
#
#             # Training loop, for each batch... Start Train !!!
#             for batch in batches:
#                 x_batch, y_batch = zip(*batch)
#                 train_step(x_batch, y_batch, train_op)
#                 current_step = tf.train.global_step(sess,global_step)
#                 if current_step % evaluate_every == 0:
#                     print('\nEvaluation:')
#                     dev_step(x_dev,y_dev,writer=dev_summary_writer)
#                     print('')
#                 if current_step % checkpoint_every == 0:
#                     path = saver.save(sess,checkpoint_prefix,global_step=current_step)
#                     print('Saved model checkpoint to {}\n'.format(path))





import  tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from TextCNN import TextCNN
from tensorflow.contrib import learn # todo learn.preprocessing.VocablaryProcessor()

# todo tf.flags.DEFINE_xxx(),用于命令行参数的传递
# todo 存在一个问题：没有模型读取的部分，当中断程序后重新运行需要从头训练


dev_sample_percentage = .1
positive_data_file = './data/rt-polarity.pos'
negative_data_file = './data/rt-polarity.neg'
embedding_dim = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
batch_size = 16
num_epochs = 10
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
allow_soft_placement = True
log_device_placement = False
filter_sizes = '3,4,5'
num_filters = 128

# Data Preparation
# ==============================================

# Load data加载数据，返回数据集和标签
print('Loading data...')
x_text, y = data_helper.load_data_and_labels(positive_data_file,negative_data_file)

# Build vocabulary 生成但是字典
# 得到最大邮件长度（单词个数），不足的用0补充
max_document_length = max([len(x.split(' ')) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))       # todo 生成word_to_id-Metrix ，不够的补零

# 数据打乱数据集
np.random.seed(32)

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffle = x[shuffle_indices]
y_shuffle = y[shuffle_indices]

# train_test_split  处理交粗糙，应当使用cross-validaton
# 从后往前取
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]

print('Vocabulary Size:{}'.format(len(vocab_processor.vocabulary_))) # todo 计算所有的词汇数量
print('Train/Dev split:{}/{}'.format(len(y_train),len(y_dev)))


def train_step(x_batch,y_batch,train_op,step):
    '''
    a single training step
    '''
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: dropout_keep_prob
    }

    _, loss, accuracy = sess.run(
        [train_op, cnn.loss, cnn.accuracy],
    feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print('{}:step {}, loss {}, acc: {}'.format(time_str,step,loss,accuracy))


def dev_step(x_batch, y_batch,step):
    '''
    evalutes model on dev set
    '''
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.dropout_keep_prob: 1.0
    }

    loss, accuracy = sess.run(
        [cnn.loss, cnn.accuracy],
        feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print('{}: step {}, loss {}, acc {}'.format(time_str,step,loss,accuracy))
    return accuracy

with tf.device('/gpu:0'):
    with tf.Graph().as_default():
        # session
        session_conf = tf.ConfigProto(
            allow_soft_placement = allow_soft_placement,
            log_device_placement = log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = TextCNN(
                sequence_length = x_train.shape[1], # 句子的长度。记住，已经填补了所有的句子的长度（我们的数据集为59）
                num_classes = y_train.shape[1],
                vocab_size = len(vocab_processor.vocabulary_),   # 邮件字典不重复单词数目
                embedding_size = embedding_dim,   # 中间层向量，也就是词嵌入的维度
                filter_sizes = list(map(int,filter_sizes.split(','))), # 卷积核大小
                num_filters = num_filters,    # 卷积核数目
                l2_reg_lambda = l2_reg_lambda) # l2惩罚项大小

            global_step = 0

            train_op = tf.train.AdamOptimizer(1e-3).minimize(cnn.loss)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            # if not os.path.exists(out_dir):
            #     os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))


            # checkpoint dirctory . tensorflow assumes the directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir,'checkpoints'))
            # checkpoint_prefix = os.path.join(checkpoint_dir,'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver()

            # write vocabulary
            vocab_processor.save(os.path.join(out_dir,'vocab'))

            # initialize all variables
            # 全局初始化
            sess.run(tf.global_variables_initializer())

            # generate batch
            batches = data_helper.batch_iter(
                list(zip(x_train,y_train)),
                batch_size,
                num_epochs
            )

            # Training loop, for each batch... Start Train !!!]
            best_acc = 0
            for batch in batches:
                global_step += 1
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, train_op,global_step)
                if global_step % evaluate_every == 0:
                    print('\nEvaluation:')
                    accurent_acc = dev_step(x_dev,y_dev,global_step)
                    print('')
                if global_step % checkpoint_every == 0 and accurent_acc > best_acc:
                    path = saver.save(sess,checkpoint_dir + '/save_net.ckpt')
                    best_acc = accurent_acc
                    print('Saved model checkpoint to {}\n'.format(path))