import tensorflow as tf
import numpy as np
from TextRNN import TextRNN
from data_util_zhihu import load_data_multilabel_new, create_voabulary, create_voabulary_label
from tflearn.data_utils import pad_sequences    # 这里使用的tflearn高级封装，可以进行数据的pad
import os
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors      # 使用加载二进制的保存文件
import pickle

num_classes = 1999
learning_rate = 0.01
batch_size = 128        # 计算机显存不够时调节小点
# todo decay参数放弃使用
decay_steps = 12000
decay_rate = 0.9
ckpt_dir = 'text_rnn_checkpoint/'
sequence_length = 100
embed_size = 100
is_training = True
num_epochs = 60
validation_every = 1
use_embedding =  True
traing_data_path = './data/train-zhihu4-only-title-all.txt'
word2vec_model_path = './data/zhihu-word2vec-title-desc.bin-100.txt'

# 1. load data(X:list of lint,y:int) 2.create seession 3.feed data 4.training (5.validation) (6.predict)

def main(_):
    # 1. load data
    if 1 == 1:
        # 1. get vocabulary of label.
        # trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_voabulary(simple='simple',word2vec_model_path=word2vec_model_path,name_scope='rnn')
        vocab_size = len(vocabulary_word2index)
        print('rnn_model.vocab_size:',vocab_size)
        vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope='rnn',voabulary_label=traing_data_path)
        train, test, _ = load_data_multilabel_new(vocabulary_word2index,vocabulary_word2index_label,multi_label_flag=False,traning_data_path=traing_data_path)
        trainX, trainY = train
        testX, testY = test

        # 2. data preprocessing.Sequence padding
        print('start padding & transform to one hot ...')
        trainX = pad_sequences(trainX,maxlen=sequence_length,value=0.0) # padding to max length
        testX = pad_sequences(testX,maxlen=sequence_length,value=0.0)

        print('trainX[0]:',trainX[0],)
        # convert labels to binary vector
        print('end padding & transform to one hot ...')

    # 2. create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 动态的分配gpu空间，需要多少占用多少
    with tf.Session(config=config) as sess:
        # instantiate mdoel
        textRNN = TextRNN(num_classes,learning_rate,batch_size,decay_steps,decay_rate,sequence_length,
                          vocab_size,embed_size,is_training)
        saver = tf.train.Saver()
        if os.path.exists(ckpt_dir + 'checkpoint'):
            print('Restoring Variables from Checkpoint for rnn model.')
            saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))  # todo 怎么找到最近保存的文件的？
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if use_embedding:   # load pre_trained word embedding
                assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textRNN,word2vec_model_path=word2vec_model_path)
        curr_epoch = sess.run(textRNN.epoch_step)

        # 3.feed data & training
        number_of_training_data = len(trainX)
        for epoch in range(curr_epoch,num_epochs):
            loss, acc, counter =  0.0, 0.0, 0
            for start, end in zip(range(0,number_of_training_data,batch_size),range(batch_size,number_of_training_data,batch_size)):
                if epoch == 0 and counter == 0:
                    print('trainX[start:end]:',trainX[start:end])
                curr_loss, curr_acc, _ = sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.train_op],feed_dict={textRNN.input_x:trainX[start:end],textRNN.input_y:trainY[start:end],
                                                                                                                  textRNN.dropout_keep_prob:1.0})
                loss, counter, acc = loss + curr_loss, counter + 1,acc + curr_acc
                if counter % 500 ==0:
                    print('Epoch {} \tBatch {}\tTrain Loss:{:.3}\tTrain Accuracy:{:.3}'.format(epoch,counter,loss/float(counter),acc/float(counter)))
            # epoch increament
            print('going to increament epoch counter ...')
            sess.run(textRNN.epoch_increament)
            # 4.validation
            print(epoch,validation_every,(epoch % validation_every == 0))
            if epoch % validation_every == 0:
                eval_loss, eval_acc = do_eval(sess,textRNN,testX,testY,batch_size,vocabulary_index2word_label)
                print('Epoch {} Validation Loss: {:.3} \tValidation Accuracy:{:.3}'.format(epoch,eval_loss,eval_acc))
                # save model to checkpoint
                save_path = ckpt_dir + 'model.ckpt'
                # saver.save(sess,save_path,global_step=epoch)
                saver.save(sess,save_path,global_step=textRNN.global_step)
        # 5. test in testData and report accuracy
        test_loss,test_acc = do_eval(sess,textRNN,testX,testY,batch_size,vocabulary_index2word_label)
    pass

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textRNN,word2vec_model_path=None):
    print('using pre-trained word embedding.started.word2vec_model_path:',word2vec_model_path)
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path,binary=True)
    # word-id 字典的生成，但是对于gensim的设计特点可以直接使用字典模式
    word_embedding_2dlist = [[]] * vocab_size # create an empty embedding lsit
    word_embedding_2dlist[0] = np.zeros(embed_size)
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # todo ?
    count_exist = 0
    count_not_exist = 0
    for i in range(1,vocab_size):   # loop each word
        word = vocabulary_index2word[i] # get a word
        embedding = None
        try:
            embedding = word2vec_model[word]
        except Exception:
            embedding = None
        if embedding is not None:   # the word exist a emebdding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1 #  assign array to this word
        else:   # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound,bound,embed_size)
            count_not_exist = count_not_exist + 1   # init a random value for the word
    word_embedding_final = np.array(word_embedding_2dlist)  # convert to 2d array
    word_embedding = tf.constant(word_embedding_final,dtype=tf.float32) # convert to tensor
    t_assign_embedding = tf.assign(textRNN.Embedding,word_embedding)
    sess.run(t_assign_embedding)
    print('word. exits embedding: {} ;word not exist embedding:{}'.format(count_exist,count_not_exist))
    print('using pre-trained word embedding.ended...')


# 在验证集上做验证，报告损失，精确度
def do_eval(sess,textRNN,evalX,evalY,batch_size,vocabulary_index2word_label):
    number_examples = len(evalX)
    print(number_examples)
    eval_loss, eval_acc, eval_counter = 0.0,0.0,0
    for start, end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, logits, curr_eval_acc = sess.run([textRNN.loss_val,textRNN.logits,textRNN.accuracy],
                                                         feed_dict={textRNN.input_x:evalX[start:end],textRNN.input_y:evalY[start:end],
                                                                    textRNN.dropout_keep_prob:1})
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss,eval_acc + curr_eval_acc, eval_counter + 1

    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)


# 从logits中提取前五
def get_label_using_logits(logits,vocabulary_index2word_label,top_number=1):
    print('get_label_using_logits.logits:',logits) # 1-d array:array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32)
    index_list = np.argsort(logits)[-top_number]
    index_list = index_list[::-1]
    return index_list

# 统计预测的准确率
def calculate_accuracy(labels_predicted,labels,eval_counter):
    label_nozero = []
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    if eval_counter < 2:
        print('labels_predicted: {} ; labels_nozero: {}'.format(labels_predicted,label_nozero))
    count = 0
    label_dict = {x:x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict,None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

if __name__ == '__main__':
    main(1)