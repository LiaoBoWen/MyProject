import tensorflow as tf
import numpy as np
from TextRNN import TextRNN
from data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences
import os
import codecs

num_classes = 1999
learning_rate = 0.01
batch_size = 80
decay_rate = 0.9
decay_steps = 12000
ckpt_dir = 'text_rnn_checkpoint/'
sequence_length = 100
embed_size = 100
is_training = False
training_data_path = './data/train-zhihu4-only-title-all.txt'
word2vec_model_path = 'zhihu-word2vec.bin-100'
predict_target_file = 'text_rnn_checkpoint/zhihu_result_rnn5'
predict_source_file = 'test-zhihu-forpredict-v4only-title.txt'


def main(_):
    # 1. load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',word2vec_model_path=word2vec_model_path,name_scope='rnn')
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope='rnn')
    questionid_question_lists = load_final_test_data(predict_source_file)
    test = load_data_predict(vocabulary_word2index,vocabulary_word2index_label,questionid_question_lists)
    testX =[]
    question_id_list = []
    for tuple in test:
        question_id,question_string_list = tuple
        question_id_list.append(question_id)
        testX.append(question_string_list)

    # 2.data preprocessing :sequence padding
    print('string padding...')
    testX2 = pad_sequences(testX,maxlen=sequence_length,value=0.)
    print('end padding...')

    # 3.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.instantiate model
        textRNN = TextRNN(num_classes,learning_rate,batch_size,decay_steps,decay_rate,sequence_length,vocab_size,embed_size,is_training)
    saver = tf.train.Saver()
    if os.path.exists(ckpt_dir + 'checkpoint'):
        print('Restore Variables from Checkpoint for TextRNN')
        saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
    else:
        print("Can't find the checkpoint.going to stop")
        return

    # 5.feed data
    number_of_training_data = len(testX2)
    print('number_of_training_data:',number_of_training_data)
    index = 0
    predict_target_file_f = codecs.open(predict_target_file,'a','utf8')
    for start, end in zip(range(0,number_of_training_data,batch_size),range(batch_size,number_of_training_data + 1,batch_size)):
        logits = sess.run(textRNN.logits,feed_dict={textRNN.input_x:testX2[start:end],textRNN.dropout_keep_prob:1}) # shape of logits: (1,1999)

        print('start: {} ;end: {}'.format(start,end))

        question_id_sublist = question_id_list[start:end]
        get_label_using_logits_batch(question_id_sublist.logits,vocabulary_index2word_label,predict_target_file_f)

        index = index + 1
    predict_target_file_f.close()

# get label using logits
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=5):
    print('get_label_using_logits.shape:',logits.shape) # (10,1999)=[batch_size,numbels] ==>need (10,5)
    index_list = np.argsort(logits)[-top_number:]
    index_list = index_list[::-1]
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        label_list.append(label)    #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    print('get_label_using_logits.label_list',label_list)
    return label_list


def get_label_using_logits_batch(question_id_sublist,logits_batch,vocabulary_index2word_label,f,top_number=5):
    for i, logits in enumerate(logits_batch):
        index_list = np.argsort(logits)[-top_number:]
        index_list = index_list[::-1]
        label_list = []
        for index in index_list:
            label = vocabulary_index2word_label[index]
            label_list.append(label)
        write_question_id_with_labels(question_id_sublist[i],label_list,f)
    f.flush()

def write_question_id_with_labels(question_id,labels_list,f):
    labels_string = ','.join(labels_list)
    f.write(question_id + ',' + labels_string + '\n')


if __name__ == '__main__':
    pass