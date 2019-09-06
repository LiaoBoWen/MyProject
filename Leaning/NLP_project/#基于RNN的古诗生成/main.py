import os
import numpy as np
import tensorflow as tf
import data_preprocessor
from gen_poetry import *
from LSTM_model import rnn_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

batch_size = 64
poetry_file = './data/poem.txt'

def train(words,poetry_vector,x_batches,y_batches):
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    output_targets = tf.placeholder(tf.int32,[batch_size,None])
    end_points= rnn_model(len(words),input_data=input_data,output_data=output_targets,batch_size=batch_size)

    saver = tf.train.Saver(tf.global_variables())

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    merge = tf.summary.merge_all()# todo ???
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('./logs',sess.graph)
        sess.run(init_op)

        start_epoch = 0
        model_dir = './model'
        epochs = 50
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            # 导出模型
            saver.restore(sess,checkpoint)
            print('## restore from the checkpoint {}'.format(checkpoint))
            # 从上次技术的地方开始继续训练
            start_epoch += int(checkpoint.split('-')[-1])
            print('## start training...')
        try:
            for epoch in range(start_epoch,epochs):
                n_chunk = len(poetry_vector) // batch_size
                for n in range(n_chunk):
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']],
                    feed_dict={input_data:x_batches[n],output_targets:y_batches[n]})
                    print('Epoch: {}, batch: {}, training loss: {}'.format(epoch,n,loss))
                    if epoch % 5 == 0:
                        saver.save(sess,os.path.join(model_dir,'poetry'),global_step=epoch)
                        result = sess.run(merge,feed_dict={input_data:x_batches[n],output_targets:y_batches[n]})
                        writer.add_summary(result,epoch * n_chunk + n)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now ...')
            saver.save(sess,os.path.join(model_dir,'pooetry'),global_step=epoch)
            print('## Last epoch were saved, next time will start form epoch {}'.format(epoch))

if __name__ == "__main__":

    word, poetry_vector, to_num, x_batches, y_batches = data_preprocessor.poetry_process()
    train(word,poetry_vector, x_batches, y_batches)

    # gen_poetry(word,to_num)
    # generate(word, to_num, style_words='狂沙将军战燕然')