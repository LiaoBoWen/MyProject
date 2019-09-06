import numpy as np
import tensorflow as tf
from LSTM_model import rnn_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def to_word(predict,words):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t,np.random.rand(1) * s))
    if sample > len(words):     # todo 这里就算出现进度丢失的情况也不会大于len(words),那到底是什么原因呢？
        sample = len(words)
    return words[sample]

def generate(words,to_num,style_word=None):
    batch_size = 1
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    cell_model = rnn_model(len(words),input_data,batch_size=batch_size)
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        check_point = tf.train.latest_checkpoint('./model')
        saver.restore(check_point)

        x = np.array(to_num('B')).reshape(1,1)

        _, last_state = sess.run([cell_model.prediction,cell_model.loss],feed_dict={input_data:x})

        if style_word:
            for i in style_word:
                x = np.array(to_num(i)).reshape(1,1)
                predict, last_state = sess.run([cell_model.prediction,cell_model.last_state],feed_dict={input_data:x,last_state:last_state})

        start_words = list('少小离家老大回')
        start_len = len(start_words)

        result = start_words.copy()
        max_len = 200
        for i in range(max_len):
            if i < start_len:
                w = start_words[i]
                x = np.array(to_num(w)).reshape(1,1)
                predict, last_state = sess.run([cell_model.prediction,cell_model.last_state],feed_dict={input_data:x,last_state:last_state})

            else:
                predict, last_state = sess.run([cell_model.prediction,cell_model.last_state],feed_dict={input_data:x,last_state:last_state})
                w = to_word(predict,words)
                x = np.array(to_num(w)).reshape(1,1)

                if w == 'E':
                    break
                result.append(w)
        print(''.join(result))

        return ''.join(result)