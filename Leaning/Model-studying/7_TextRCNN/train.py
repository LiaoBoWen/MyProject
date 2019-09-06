import tensorflow as tf
from TextRCNN import TextRCNN
from config import config
import data_helper
import time
import datetime
import os

def train(config):
    print('parameters:')
    print(config)

    # load data
    print('load data')
    X, y = data_helper.process_data(config)    # X=[[seq1],[seq2]]   y=[,,,,]

    # make vocab
    print('make vocab...')
    word2index, label2index = data_helper.generate_vocab(X,y,config)

    # padding data
    print('padding data')
    input_x, input_y = data_helper.padding(X,y,config,word2index,label2index)

    # split data
    print('split data...')
    x_train, y_train, x_test, y_test, x_dev, y_dev = data_helper.split_data(input_x,input_y,config)

    print('length train: {}'.format(len(x_train)))
    print('length test: {}'.format(len(x_test)))
    print('length dev: {}'.format(len(x_dev)))

    print('training...')

    with tf.Graph().as_default():
        sess_config = tf.ConfigProto(
            allow_soft_placement=config['allow_soft_placement'],
            log_device_placement=config['log_device_placement']
        )
        with tf.Session(config=sess_config) as sess:
            rcnn = TextRCNN(config)

        # training procedure
        global_step = tf.Variable(0,name='globel_step',trainable=False)
        train_op = tf.train.AdamOptimizer(config['learning_rate']).minimize(rcnn.loss,global_step=global_step)

        # output dir for models
        timestamp = str(int(time.time()))
        outdir = os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))
        if not os.path.exists(os.path.join(os.path.curdir,'runs')):
            os.mkdir(os.path.join(os.path.curdir,'runs'))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        print('writing to {}'.format(outdir))

        # checkpoint dictory
        checkpoint_dir = os.path.abspath(os.path.join(outdir,'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir,'model')

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=config['num_checkpoints'])

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch,y_batch):
            feed_dict = {
                rcnn.input_x:x_batch,
                rcnn.input_y:y_batch,
                rcnn.dropout_keep_prob:config['dropout_keep_prob']
            }

            _, step, loss, accuracy = sess.run(
                [train_op, global_step, rcnn.loss, rcnn.accuracy],
                feed_dict=feed_dict
            )

            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {}， acc {}'.format(time_str,step,loss,accuracy))

        def dev_step(x_batch, y_batch):
            feed_dict = {
                rcnn.input_x:x_batch,
                rcnn.input_y:y_batch,
                rcnn.dropout_keep_prob:1.0
            }

            step,loss, accuracy = sess.run(
                [global_step, rcnn.loss, rcnn.accuracy],
                feed_dict=feed_dict
            )

            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {}, acc {}'.format(time_str,step,loss,accuracy))
        # generate batches
        batches = data_helper.generate_batchs(x_train,y_train,config)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            print(y_batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess,global_step)
            if current_step % config['evaluate_every'] == 0:
                print('Evaluation:')
                dev_step(x_dev,y_dev)

            if current_step % config['checkpoint_every'] == 0:
                path =  saver.save(sess,checkpoint_prefix,global_step=current_step)
                print('save model checkpoint to {}'.format(path))

        # test accuracy
        test_accuracy = sess.run([rcnn.accuracy],feed_dict={
            rcnn.input_x:x_test,rcnn.input_y:y_test,rcnn.dropout_keep_prob:1.0
        })
        print('Test dataset accuracy: {}'.format(test_accuracy))

if __name__ == '__main__':
    train(config)