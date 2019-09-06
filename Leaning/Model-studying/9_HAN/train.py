import tensorflow as tf
import numpy as np
import time
import os
import pickle
from utils import prjPaths,get_logger
from HAN_model_1 import HAN
from dataProcessing import IMDB

CONFIG = {
    'dataset':'imdb',
    'run_type':'train',
    'embedding_dim':300,
    'batch_size':256,
    'num_epochs':25,
    'evaluate_every':100,
    'log_summaries_every':30,
    'checkpoint_every':100,
    'num_checkpoint':5,
    'max_grad_norm':5.,
    'dropout_keep_proba':0.5,
    'learning_rate':1e-3,
    'per_process_gpu_memory_fraction':0.9
}

def train():
    paths = prjPaths()

    with open(os.path.join(paths.LIB_DIR,CONFIG['dataset'],'persisted_vars.p'),'rb') as handle:
        persisted_vars = pickle.load(handle)

    persisted_vars['embedding_dim'] = CONFIG['embedding_dim']
    persisted_vars['max_grad_norm'] = CONFIG['max_grad_norm']
    persisted_vars['dropout_keep_proba'] = CONFIG['dropout_keep_proba']
    persisted_vars['learning_rate'] = CONFIG['learning_rate']
    pickle._dump(persisted_vars,open(os.path.join(paths.LIB_DIR,CONFIG['dataset'],'persisted_vars.p'),'wb'))

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=CONFIG['per_process_gpu_memory_fraction'])
        session_conf =  tf.ConfigProto(allow_soft_placement=True,
                                       log_device_placement=False,
                                       gpu_options=gpu_options)

        session_conf.gpu_options.allocator = 'BFC'

        with tf.Session(config=session_conf) as sess:
            han = HAN(max_seq_len=persisted_vars['max_grad_norm'],
                      max_sent_len=persisted_vars['max_sent_len'],
                      num_classes=persisted_vars['num_classes'],
                      vocab_size=persisted_vars['vocab_size'],
                      embedding_size=persisted_vars['embedding_dim'],
                      max_grad_norm=persisted_vars['max_grad_norm'],
                      dropout_keep_proba=persisted_vars['dropout_keep_proba'],
                      learning_rate=persisted_vars['learning_rate']
                      )

            global_step = tf.Variable(0,name='global_step',trainable=False)

            # 梯度裁剪需要获取训练参数
            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(han.loss,tvars),
                                                        han.max_grad_norm)

            optimizer = tf.train.AdamOptimizer(han.learning_rate)  # todo 尝试其他参数

            train_op =  optimizer.apply_gradients(zip(grads,tvars),
                                                    name='train_op',
                                                  global_step=global_step)

            merge_summary_op = tf.summary.merge_all()
            train_summary_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR,CONFIG['run_type']),sess.graph)

            # todo 这里的保存对象换成sess
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=CONFIG['num_checkpoint'])

            sess.run(tf.global_variables_initializer())

            # _________train__________
            def train_step(epoch,x_batch,y_batch,docsize,sent_size,is_training):
                tic = time.time()

                feed_dict = {han.input_x:x_batch,
                             han.input_y:y_batch,
                             han.sentence_lengths:docsize,
                             han.word_legths:sent_size,
                             han.sis_training:is_training}
                _, step, loss, accuracy, summaries = sess.run([train_op,global_step,han.loss,han.accuracy,merge_summary_op],feed_dict=feed_dict)

                time_elapsed = time.time() - tic

                if is_training:
                    print('Training||CurrentEpoch: {} || GlobalStep: {} || ({} sec/sep) || Loss {:g}) || Accuracy {:g}'.format(epoch + 1, step, time_elapsed, loss, accuracy))

                if step % CONFIG['log_summaries_every'] == 0:
                    train_summary_writer.add_summary(summaries,step)
                    print(f'Saved model summaries to {os.path.join(paths.SUMMARY_DIR,CONFIG["run_type"])} \n')

                if step % CONFIG['checkpoint_every'] == 0:
                    chkpt_path = saver.save(sess,os.path.join(paths.CHECKPOINT_DIR,'han'),
                                            global_step=step)
                    print('Saved model checkpoint to {} \n'.format(chkpt_path))

            imdb = IMDB(action='fetch')
            x_train, y_train, docsize_train, sent_size_train = imdb.get_data(type=CONFIG['run_type'])

            for epoch, batch in imdb.get_batch(data=list(zip(x_train,y_train,docsize_train,sent_size_train)),
                                               batch_size=CONFIG['batch_size'],
                                               num_epoch=CONFIG['num_epochs']):
                x_batch, y_batch, docsize, sent_size = zip(*batch)

                train_step(epoch=epoch,
                           x_batch=x_batch,
                           y_batch=y_batch,
                           docsize=docsize,
                           sent_size=sent_size,
                           is_training=True)