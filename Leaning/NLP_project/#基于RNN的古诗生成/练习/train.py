import pickle
import os

from data_preprocessor import get_batches, poetry_process
from LSTM_model import rnn_model
from gen_poetry import generate
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

text_path = '../data/poems.txt'
batch_size = 64

def train(words,poetry_vector,word_num_map,batch_size=64,epoches=50):
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    output_data = tf.placeholder(tf.int32,[batch_size,None])


    cell_model = rnn_model(len(words),input_data=input_data,output_data=output_data,batch_size=batch_size)
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    merge = tf.summary.merge_all()
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('./logs',sess.graph)
        sess.run(init_op)
        start_epoch = 0
        model_dir = './model'
        checkpoint = tf.train.latest_checkpoint(model_dir)

        if checkpoint:
            saver.restore(sess,checkpoint)
            print('## restore model from {}'.format(checkpoint))
            start_epoch = int(checkpoint.split('-')[-1])
            print('## start training ...')
        try:
            for epoch in range(start_epoch, epoches):
                n_bucket = len(poetry_vector) // batch_size

                for n, (x_batch, y_batch) in enumerate(get_batches(poetry_vector,word_num_map,batch_size=batch_size)):
                    loss, _, _ = sess.run([cell_model.total_loss,cell_model.last_state,cell_model.train_op],feed_dict={input_data:x_batch,output_data:y_batch})
                    print('Epoch: {} batch: {} training loss: {}'.format(epoch,n,loss))
                    if n % 5 == 0:
                        saver.save(sess,os.path.join(model_dir,'poetry'),global_step=epoch)
                        result = sess.run(merge,feed_dict={input_data:x_batch,output_data:y_batch})
                        writer.add_summary(result,epoch * n_bucket + n)
        except KeyboardInterrupt:
            print('## ERROR:Interrupt !!! try save model now...')
            saver.save(sess,os.path.join(model_dir,'poetry'),global_step=epoch)
            print('## Last epoch model were saved, next time will train form epoch {}'.format(epoch))

if __name__ == '__main__':
    words, poetry_vector, to_num, word_num_map = poetry_process()
    train(words=words,poetry_vector=poetry_vector,word_num_map=word_num_map,batch_size=batch_size,epoches=50)

    # generate()