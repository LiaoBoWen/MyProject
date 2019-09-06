import tensorflow as tf
import os
import math
import logging
from module import *
from data_preprocess import *
from tqdm import tqdm
from model import Transformer
from utils import save_hparams, save_variable_specs, get_hypotheses,calc_bleu
from Params import params

path = '../data'
vocab_path = './data/vocab.txt'

logging.basicConfig(level=logging.INFO)

logging.info('# hparams')
Params = params()
parser = Params.parser
hp = parser.parse_args()
save_hparams(hp,hp.logdir)

logging.info('# Prepare train/eval batches')
train_batches, num_train_batches, num_train_samples = get_batch(path,vocab_path,hp.batch_size,shuffle=True)

iter = tf.data.Iterator.from_structure(train_batches.output_types,train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)

logging.info('# Load model')
m =  Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs,ys)

logging.info('# Session')
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info('Initializing from scratch')
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess,ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir,sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs.total_steps + 1)):
        _, _gs, _summary = sess.run([train_op,global_step,train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary._gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info('epoch {} is done'.format(epoch))
            _loss = sess.run(loss)

            model_output = '2019_E{}L{}'.format(epoch,_loss)
            print('epoch: {:3d}, loss: {:.2f}'.format(epoch,_loss))
            if not os.path.exists(hp.evaldir):
                os.mkdir(hp.evaldir)
            logging.info('# save models')
            ckpt_name = os.path.join(hp.logdir,model_output)
            saver.save(sess,ckpt_name,global_step=_gs)
            logging.info('after training of {} epochs, {} has been saved.'.format(epoch,ckpt_name))

            logging.info('# fall back to train mode')
            sess.run(train_init_op)
    summary_writer.close()

logging.info('Finished.')