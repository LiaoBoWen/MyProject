import numpy as np
import tensorflow as tf

from LSTM_model import rnn_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess =tf.Session(config=config)

def to_word(predict,vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t,np.random.rand(1) * s)) # todo 干嘛的????
    if sample > len(vocabs):    # 因为会出现精度丢失的情况
        sample = len(vocabs)
    return vocabs[sample]   # [np.argmax(predict]

def gen_poetry(words,to_num):
    batch_size = 1
    print('模型保存目录为:{}'.format('./model'))
    input_data =tf.placeholder(tf.int32,[batch_size,None])
    end_points = rnn_model(len(words),input_data=input_data,batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables()) # todo 全局变量?
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()) # todo 为什么这里用到了本地变量？

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess,checkpoint)

        x = np.array(to_num('B')).reshape(1,1)

        _, last_state = sess.run([end_points['prediction'],end_points['last_state']],feed_dict={input_data:x})

        word = input('请输入起始字符: ')
        poem_ = ''
        while word != 'E':
            poem_ += word
            x = np.array(to_num(word)).reshape(1,1)
            predict, last_state = sess.run([end_points['prediction'],end_points['last_state']],feed_dict={input_data:x,end_points['initial_state']:last_state})

            word = to_num(predict,words)
        print(poem_)
        return poem_

def generate(words,to_num,style_words='狂沙将军战燕然，大漠孤烟黄河骑。'):
    batch_size = 1
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    end_point = rnn_model(len(words),input_data=input_data,batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # checkpoint = tf.train.latest_checkpoint('./model')
        # saver.restore(sess,checkpoint)
        saver.restore(sess,'./model/poetry-0')

        x = np.array(to_num('B')).reshape(1,1)
        # 同时获取last_state用于下一个的预测
        _, last_state = sess.run([end_point['prediction'],end_point['last_state']],feed_dict={input_data:x})


        if style_words:
            # 计算style的state，相当于给出start_words和这句然后生成下面的诗词
            for word in style_words:
                x = np.array(to_num(word)).reshape(1,1)
                last_state = sess.run(end_point['last_state'],feed_dict={input_data:x,end_point['initial_state']:last_state})

        start_words = list('少小离家老大回')
        # start_words = list(input('请输入起始语句：'))
        start_words_len = len(start_words)

        result = start_words.copy()
        max_len = 200
        for i in range(max_len):
            # 这里不需要输出，但是我们需要计算出state用于这局之后的生成
            if i < start_words_len:
                w = start_words[i]
                x = np.array(to_num(w)).reshape(1,1)
                predict, last = sess.run([end_point['prediction'],end_point['last_state']],feed_dict={input_data:x,end_point['initial_state']:last_state})
            else:
                predict, last_state = sess.run([end_point['prediction'],end_point['initial_state']],feed_dict={input_data:x,end_point['initial_state']:last_state})
                w = to_word(predict,words)
                # w = words[np.argmax(predict)]
                x = np.array(to_num(w)).reshape(1,1)
                if w == 'E':
                    break
                result.append(w)
        print(''.join(result))