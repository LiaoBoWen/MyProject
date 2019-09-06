import os
import tqdm
import time
import pickle
import numpy as np
from scipy import stats     # todo stats
from collections import Counter
from utils import prjPaths, get_logger

from dataProcessing import IMDB
from HAN_model_1 import HAN
import tensorflow as tf


CONFIG = {
    'dataset':'imdb',
    'run_type':'val',
    'log_summaries_every':30,
    'per_process_gpu_memory_fraction':0.9,
    'wait_for_checkpoint_files':False
}

def get_most_recently_create_file(files):
    return max(files,key=os.path.getctime)

def test():
    MINUTE = 60
    paths = prjPaths()
    print('loading persisted variables...')
    with open(os.path.join(paths.LIB_DIR,CONFIG['dataset'],'persisted_vars.p'),'rb') as handle:
        persisted_vars = pickle.load(handle)

    graph = tf.Graph()
    with graph.as_default():
        # Set GPU options
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=CONFIG['per_process_gpu_memory_fraction'])
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False,
                                      gpu_options=gpu_options)

        session_conf.gpu_options.allocator_type = 'BFC'


        with tf.Session(config=session_conf) as sess:
            # Insert model
            han = HAN(max_seq_len=persisted_vars['max_seq_len'],
                      max_sent_len=persisted_vars['max_sent_len'],
                      num_classes=persisted_vars['num_classes'],
                      vocab_size=persisted_vars['vocab_size'],
                      embedding_size=persisted_vars['embedding_dim'],
                      max_grad_norm=persisted_vars['max_grad_norm'],
                      dropout_keep_proba=persisted_vars['dropout_keep_proba'],
                      learning_rate=persisted_vars['learning_rate'])

            global_step = tf.Variable(0,name='global_step',trainable=False)
            tvars = tf.trainable_variables()

            # todo 这个方法返回的是什么
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(han.loss,tvars),
                                                        han.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(han.learning_rate)
            test_op = optimizer.apply_gradients(zip(grads,tvars),
                                                name=f'{CONFIIG["run_type"]}_op',
                                                global_step=global_step)

            merge_summary_op = tf.summary.merge_all()
            test_summary_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR,CONFIG['run_type']),sess.graph)

            meta_file = get_most_recently_create_file([os.path.join(paths.CHECKPOINT_DIR,file) for file in os.listdir(paths.CHECKPOINT_DIR) if file.endswith('.meta')])
            saver = tf.train.import_meta_graph(meta_file)

            sess.run(tf.global_variables_initializer())

            def test_step(sample_num,x_batch,y_batch,docsize,sent_size,is_training):
                feed_dict = {han.input_x:x_batch,
                             han.input_y:y_batch,
                             han.sentence_lengths:docsize,
                             han.word_lengths:sent_size,
                             han.is_training:is_training}
                loss, accuracy = sess.run([han.loss,han.accuracy],feed_dict=feed_dict)
                return loss, accuracy

            if CONFIG['dataset'] == 'imdb':
                dataset_controller = IMDB(action='fetch')
            else:
                exit('set dataset flag to appropiate dataset')

            x, y, docsize, sent_size = dataset_controller.get_data(type=CONFIG['run_key'])
            all_evaluated_chkpts = []

            while True:
                if CONFIG['wait_for_checkpoint_files']:
                    time.sleep(2 * MINUTE)  # wait for create new checkpoint file
                else:
                    time.sleep(0 * MINUTE)

                if tf.train.latest_checkpoint(paths.CHECKPOINT_DIR) in all_evaluated_chkpts:
                    continue

                saver.restore(sess,tf.train.latest_checkpoint(paths.CHECKPOINT_DIR))
                all_evaluated_chkpts.append(tf.train.latest_checkpoint(paths.CHECKPOINT_DIR))

                losses = []
                accuracies = []

                tic = time.time()

                for i,batch in enumerate(tqdm(list(zip(x, y, docsize, sent_size)))):
                    x_batch, y_batch, docsize_batch, sent_size_batch = batch
                    x_batch = np.expand_dims(x_batch,axis=0)
                    y_batch = np.expand_dims(y_batch,axis=0)
                    sent_size_batch = np.expand_dims(sent_size_batch,axis=0)

                    loss, accuracy = test_step(sample_num=1,
                                               x_batch=x_batch,
                                               y_batch=y_batch,
                                               docsize=docsize,
                                               sent_size=sent_size,
                                               is_training=False)

                    losses.append(loss)
                    accuracies.append(accuracy)

                time_elapsed = time.time() - tic
                losses_accuracy_vars = {'losses':losses,'accuracy':accuracies}

                print('Time taken to complete {} evaluate of {} checkpoint : {}'.format(CONFIG['run_type'],all_evaluated_chkpts[-1],time_elapsed))

                for k in losses_accuracy_vars.keys():
                    print('stats for {}:{}'.format(k,stats.describe(losses_accuracy_vars[k])))
                    print(Counter(losses_accuracy_vars[k]))

                filename, ext = os.path.splitext(all_evaluated_chkpts[-1])
                pickle._dump(losses_accuracy_vars,open(os.path.join(paths.LIB_DIR,CONFIG['dataset'],'losses_accuracies_vars_{}.p'.format(filename.split('/')[-1])),'wb'))