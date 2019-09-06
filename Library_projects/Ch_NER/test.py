import os
from CRF_bilstm import CRF_bilstm
from config import config
from process_data import get_word2id,generate_batch
import tensorflow as tf


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
            inputs = input('Please input your sentence:')
            if inputs == '' or inputs.isspace():
                print('Finish.')
                break
            inputs = [(inputs,['O'] * len(inputs))]
            for sent, _, sent_len, _ in generate_batch(inputs,word2id,1,1):
                pred_labels = model.dev_decode(sess, sent, sent_len)
                ORG_lst, LOC_lst, PER_lst = model.precision_and_recall_and_fb1(pred_labels)
                for dev_seq, idxs in zip(sent, ORG_lst):
                    for idx in idxs:
                        print(''.join(map(lambda i: id2word[dev_seq[i]], idx)))