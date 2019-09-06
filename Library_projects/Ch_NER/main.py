import re
import os
import time
from config import config
from CRF_bilstm import CRF_bilstm
from process_data import read_corpus,get_word2id,generate_batch
import tensorflow as tf

if __name__ == '__main__================================':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train_data = read_corpus(config['train_path'])
    test_data = read_corpus(config['test_path'])
    word2id, id2word = get_word2id(config['word2id_path'])
    print('word2id size: {}'.format(len(word2id)))
    vocab_size = len(word2id)
    # outdir = '{}{}'.format(config['save_path'],int(time.time()))
    outdir = '{}{}'.format(config['save_path'],1564541944)
    summary_path = os.path.join(outdir,'summary')
    checkpoint_path = os.path.join(outdir,'checkpoint/')


    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8





    with tf.Session(config=session_config) as sess:
        model = CRF_bilstm(vocab_size,config['num_tags'],config['learning_rate'],
                           config['hidden_unit'],config['clip_grad'])

        saver = tf.train.Saver(tf.global_variables())
        if os.path.exists(checkpoint_path):
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
            saver.restore(sess,checkpoint_file)
            print('Restored model from Checkpoint file.')
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialed the sess.')

        file_writer = tf.summary.FileWriter(summary_path)

        for sent_batch, labels_batch, lens , epoch_id in generate_batch(train_data,
                                                             word2id,
                                                             config['epoch'],
                                                             config['batch_size'],
                                                             shuffle=True):
            feed_dict = {model.inputs:sent_batch,
                         model.labels:labels_batch,
                         model.seq_len:lens,
                         model.dropout_prob:config['dropout_prob'],
                         }
            _, loss, summary, global_step = sess.run([model.train_op, model.loss, model.merged, model.global_step],
                                                     feed_dict=feed_dict)

            if global_step % 300 == 0:
                print('Epoch:{:<3} Global_step:{:<5} Loss:{}'.format(epoch_id + 1, global_step, loss))
                saver.save(sess,checkpoint_path + 'check',global_step=model.global_step)

            file_writer.add_summary(summary,global_step)

            if global_step % 100 == 0:
                print('================ Dev ==================')
                for dev, dev_labels, dev_len, _ in generate_batch(test_data,word2id,1,len(test_data)):
                    pred_labels, loss = model.dev_decode(sess,dev,dev_len,dev_labels)
                    print(pred_labels)
                    ORG_lst, LOC_lst, PER_lst = model.precision_and_recall_and_fb1(dev_labels)

                    print('Dev-Loss:{}\n'.format(loss))
                    for dev_seq, idxs in zip(dev,ORG_lst):
                        for idx in idxs:
                            print(''.join(map(lambda i:id2word[dev_seq[i]],idx)))



if __name__ == '__main__':
    word2id, id2word = get_word2id('/media/liao/Data/My_Projects/Library_projects/Ch_NER/data/word2id.pkl',
                                   '/media/liao/Data/My_Projects/Library_projects/Ch_NER/data/id2word.pkl')
    outdir = '{}{}'.format(config['save_path'], 1564541944)
    checkpoint_path = os.path.join(outdir, 'checkpoint/')

    model = CRF_bilstm(len(word2id),config['num_tags'],config['learning_rate'],
                       config['hidden_unit'],config['clip_grad'])
    with tf.Session() as sess:
        saver = tf.train.Saver()
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
        saver.restore(sess,checkpoint_file)
        while True:
            inputs_ = input('Please input your sentence:')
            if inputs_ == '' or inputs_.isspace():
                print('Finish.')
                break
            inputs = [(inputs_,['O'] * len(inputs_))]
            for sent, _, sent_len, _ in generate_batch(inputs,word2id,1,1):
                pred_labels = model.dev_decode(sess, sent, sent_len)
                ORG_lst, LOC_lst, PER_lst = model.precision_and_recall_and_fb1(pred_labels)
                ORGs, LOCs, PERs = [], [], []
                for org_idxs, loc_idxs, per_idxs in zip(ORG_lst,LOC_lst,PER_lst):
                    for idx in org_idxs:
                        ORGs.append(''.join(map(lambda i: inputs_[i], idx)))
                    for idx in loc_idxs:
                        LOCs.append(''.join(map(lambda i: inputs_[i], idx)))
                    for idx in per_idxs:
                        PERs.append(''.join(map(lambda i: inputs_[i], idx)))
                print('ORG: {}'.format(set(ORGs)))
                print('LOC: {}'.format(set(LOCs)))
                print('PER: {}'.format(set(PERs)))