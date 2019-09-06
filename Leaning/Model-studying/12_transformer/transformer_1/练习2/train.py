import os
import logging
from tqdm import tqdm
import tensorflow as tf

from utils import *
from Model import Transformer
from Hyparams import hyparams as hp
from Data_process import load_labels
logging.basicConfig(level=logging.INFO)


path = './data'
vocab_path = './data/vocab.txt'

logging.info('# Generate data .')
train_batches, num_train_batches, num_train_samples = get_batches(path, vocab_path, hp.batch_size, shuffle=True)

iter = tf.data.Iterator.from_structure(train_batches.output_types,train_batches.output_shapes)
xs, ys = iter.get_next()
train_init_op = iter.make_initializer(train_batches)

logging.info('# Load model .')
model = Transformer(hp)
loss, train_op, global_step, train_summaries = model.train(xs,ys)


logging.info('# Session')
saver = tf.train.Saver(max_to_keep=3)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info('# Initialize model .')
        sess.run(tf.global_variables_initializer())
        save_variable_space(os.path.join(hp.logdir,'spaces'))

    else:
        logging.info('# Go on training the model .')
        saver.restore(sess,ckpt)

    # net可视化
    summaries_writer = tf.summary.FileWriter(hp.logdir,sess.graph)

    sess.run(train_init_op)
    _gs = sess.run(global_step)
    total_step = hp.num_epoches * num_train_batches

    for i in tqdm(range(_gs,total_step + 1)):
        _, _gs, _summaries = sess.run([train_op, loss, train_summaries])
        summaries_writer.add_summary(_summaries)

        if _gs and _gs % num_train_batches == 0:
            epoch = _gs / num_train_batches
            _loss = sess.run(loss)
            logging.info('epoch {} is done, loss: {}'.format(epoch,_loss))

            model_output = 'lbwE{:3d}L{:.3f}'.format(epoch,_loss)
            if os.path.exists(hp.evadir):
                os.makedirs(hp.evaldir)
            logging.info('# Save model .')
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess,ckpt_name,global_step=_gs)

            logging.info('# Fall back to train mode .')
            sess.run(train_init_op)

    summaries_writer.close()

logging.info('# Done ')