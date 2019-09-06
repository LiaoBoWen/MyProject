from keras.layers import Input,LSTM,Dense,Dropout,CuDNNLSTM
from keras.models import Model,load_model
from keras.optimizers import Adam,RMSprop
# from keras.utils import plot_model
import keras.backend.tensorflow_backend as KTF

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

import pickle
import numpy as np
import pandas as pd
from config import CONFIG
from string import punctuation
# import os

# ——————————————————————————————————————————Params———————————————————————————————————————————————
N_UNITS = 256
BATCH_SIZE = 126
EPOCH = 200
NUM_SAMPLES = 2048
english_to_chinese = True
continue_train = False

def run(english_to_chinese):
    # ———————————————————————————————————————————————————model—————————————————————————————————————————————————————————
    def model(n_input,n_output,n_unit):
        '''
        ————————————————基于字符级别的计算——————————————
        train three model!!
        表面训练的是model，但是最后真正用到的是encoder_infer和decoder_infer,其实就是训练encoder和decoder的LSTM层，只是decoder训练的时候需要进行dense与target计算误差;
        都是使用了[h,c]而不是一个单独的h;
        其实就是在训练一个模型,后面的两个模型的网络层都在第一个模型里面包含了;
        因为后面两个模型的神经层都在第一个模型里面了，所以训练第一个模型的时候就是在训练后面的两个模型;
        '''
        # Encoding 只使用了最后于的一个时间步来进行计算
        encoder_input = Input(shape=[None,n_input])
        encoder = LSTM(n_unit,return_state=True)

        # Start encode      这里使用了c值来进行计算
        _, encoder_h, encoder_c = encoder(encoder_input)

        encoder_state = [encoder_h,encoder_c]       # 以LSTM需要的state形式进行组合 => [h,c]

        # Decodeing  使用了每一个时间步来进行计算
        decoder_input = Input(shape=[None,n_output])
        decoder = LSTM(n_unit,return_state=True,return_sequences=True)

        # Start decode  用了encoding 的最后一个时间步来进行初始化
        decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
        decoder_dense = Dense(n_output,activation='softmax') # 这里进行了dense进行还原层开始的输出进行计算损失
        decoder_output = decoder_dense(decoder_output)

        model = Model([encoder_input,decoder_input],decoder_output) # 输入有两个，以列表的形式传递进去，分别是Input_1, Input_2
        encoder_infer = Model(encoder_input,encoder_state)  # 得到一个由encoder_input到encoder_state的模型，以便保存

        decoder_state_input_h = Input(shape=(n_unit,))  # 这个的输入在predict的时候进行传入
        decoder_state_input_c = Input(shape=(n_unit,))
        decoder_state_input = [decoder_state_input_h,decoder_state_input_c]

        decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
        decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]
        decoder_infer_output = decoder_dense(decoder_infer_output)

        # Model的输入需要的时Input，这里不是要维度对应，只是输入是多个而已
        decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)
        return model, encoder_infer, decoder_infer

    # ——————————————————————————————————————————model———————————————————————————————————————————————————
    def get_model(n_input,n_output,n_unit):
        encoder_input = Input(shape=[None,n_input])
        encoder = CuDNNLSTM(n_unit,return_state=True)

        _, encoder_state_h, encoder_state_c = encoder(encoder_input)
        encoder_state = [encoder_state_h,encoder_state_c]

        decoder_input = Input(shape=[None,n_output])
        decoder = CuDNNLSTM(n_unit,return_sequences=True,return_state=True)
        decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
        decoder_dense = Dense(n_output,activation='softmax')
        # decoder_output = Dropout(0.5)(decoder_output)
        decoder_output = decoder_dense(decoder_output)

        decoder_state_input_h = Input(shape=[n_unit,])
        decoder_state_input_c = Input(shape=[n_unit,])
        decoder_state_input = [decoder_state_input_h,decoder_state_input_c]     # 以encoder_state作为这里的输入

        decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
        decoder_infer_state = [decoder_infer_state_h,decoder_infer_state_c]
        decoder_infer_output = decoder_dense(decoder_infer_output)

        model = Model([encoder_input,decoder_input],decoder_output)
        encoder_infer = Model(encoder_input,encoder_state)
        decoder_infer = Model([decoder_input] + decoder_state_input,[decoder_infer_output] + decoder_infer_state)

        return model, encoder_infer, decoder_infer


    # ——————————————————————————————————data preprocess——————————————————————————————————————————————
    data_path = './data/translate2048.txt'
    df = pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,:,]
    df.replace('[,.!?，。！？]','',regex=True,inplace=True)   # 训练数据的处理方法和预测的处理方法必须一样，不能多不能少处理字符，不然基于字符的预测后面可能会出现维度问题
    df[0] = df[0].apply(lambda x: x.lower())

    if english_to_chinese:
        df.columns = ['inputs','targets']
    else:
        df.columns = ['targets','inputs']

    df['targets'] = df['targets'].apply(lambda x:'\t' + x + '\n')  # 对输出的句子做处理,因为预测的时候有必要用到\n判断是否停止，\t表示句子开头

    input_texts = df.inputs.values.tolist()
    target_texts = df.targets.values.tolist()

    # ———————————————————————————————————统计字符长度————————————————————————————————————————————————
    input_charactors = sorted(list(set(df.inputs.unique().sum()))) # todo 把所有的字符都加起来
    target_charactors = sorted(list(set(df.targets.unique().sum())))


    # ———————————————————————————————找出最长的部分作为输入输出的长度——————————————————————————————————
    if english_to_chinese:
        model_name = './model/en2zh_model.h5'
        encoder_name = './model/en2zh_encoder.h5'
        decoder_name = './model/en2zh_decoder.h5'
        INPUT_FEATURE_LENGTH = CONFIG['EN_feature_len']
        OUTPUT_FEATURE_LENGTH = CONFIG['CH_feature_len']
        INPUT_LENGTH = CONFIG['EN_len']
        OUTPUT_LENGTH = CONFIG['CH_len']
    else:
        model_name = './model/zh2en_model.h5'
        encoder_name = './model/zh2en_encoder.h5'
        decoder_name = './model/zh2en_decoder.h5'
        INPUT_FEATURE_LENGTH = CONFIG['CH_feature_len']
        OUTPUT_FEATURE_LENGTH = CONFIG['EN_feature_len']
        INPUT_LENGTH = CONFIG['CH_len']
        OUTPUT_LENGTH = CONFIG['EN_len']


    # INPUT_LENGTH = max([len(i) for i in input_texts])
    # OUTPUT_LENGTH = max([len(i) for i in target_texts])
    # INPUT_FEATURE_LENGTH = len(input_charactors)
    # OUTPUT_FEATURE_LENGTH = len(target_charactors)
    #
    # print(INPUT_LENGTH)
    # print(OUTPUT_LENGTH)+
    # print(INPUT_FEATURE_LENGTH)
    # print(OUTPUT_FEATURE_LENGTH)


    # todo 输入和输出的长度不一致
    encoder_input = np.zeros((NUM_SAMPLES,INPUT_LENGTH,INPUT_FEATURE_LENGTH))
    decoder_input = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
    decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))

    # todo char2id id2char
    input_dict = {c:k for k,c in enumerate(input_charactors)}
    input_dict_reverse = dict(zip(input_dict.values(),input_dict.keys()))
    target_dict = {c:k for k,c in enumerate(target_charactors)}
    target_dict_reverse = dict(zip(target_dict.values(),target_dict.keys()))

    pickle.dump(input_dict,open('./model/input_dict{}.pkl'.format(english_to_chinese),'wb'))
    pickle.dump(input_dict_reverse,open('./model/input_dict_reverse{}.pkl'.format(english_to_chinese),'wb'))
    pickle.dump(target_dict,open('./model/target_dict{}.pkl'.format(english_to_chinese),'wb'))
    pickle.dump(target_dict_reverse,open('./model/target_dict_reverse{}.pkl'.format(english_to_chinese),'wb'))

    for seq_index,seq in enumerate(input_texts):
        for char_index, char in enumerate(seq):
            encoder_input[seq_index,char_index,input_dict[char]] = 1

    for seq_index,seq in enumerate(target_texts):
        for char_index,char in enumerate(seq):
            decoder_input[seq_index,char_index,target_dict[char]] = 1.0
            if char_index > 0:
                decoder_output[seq_index,char_index-1,target_dict[char]] = 1.0

    # ——————————————————————————————————是否加载model继续训练———————————————————————————————————————
    if continue_train:
        model_train = load_model(model_name)
        encoder_infer = load_model(encoder_name)
        decoder_infer = load_model(decoder_name)
    else:
        model_train, encoder_infer, decoder_infer = get_model(INPUT_FEATURE_LENGTH,OUTPUT_FEATURE_LENGTH,N_UNITS)

    # 查看模型结构
    # plot_model(to_file='./model.png',model=model_train,show_shapes=True)
    # plot_model(to_file='./encoder.png',model=encoder_infer,show_shapes=True,show_layer_names=True)
    # plot_model(to_file='./decoder.png',model=decoder_infer,show_shapes=True,show_layer_names=True)

    # ———————————————————————————————————————编译、输出模型结构————————————————————————————————————
    # 这里的模型的使用RMSprop算法效果比默认参数的Adam算法的效果好
    # model_train.compile(optimizer='RMSprop',loss='categorical_crossentropy')
    op = Adam(lr=1e-3)
    model_train.compile(optimizer=op,loss='categorical_crossentropy')

    model_train.summary()
    encoder_infer.summary()
    decoder_infer.summary()

    # —————————————————————————————————————————predict———————————————————————————————————————————
    def predict_chinese(source,encoder_inference,decoder_inference,n_steps,features):

        # todo 得到enccodere的state
        state = encoder_inference.predict(source)

        predict_seq = np.zeros([1,1,features])
        predict_seq[0,0,target_dict['\t']] = 1  # 因为这里数据处理的时候我们在输出的开头部分添加了\t，所以我们这里也要进行这个操作
        output = ''
        for i in range(n_steps):
            yhat, h, c = decoder_inference.predict([predict_seq] + state)  # todo yhat的shape
            char_index = np.argmax(yhat[0,-1,:])
            char = target_dict_reverse[char_index]
            output += char
            state = [h,c]
            predict_seq = np.zeros([1,1,features])
            predict_seq[0,0,char_index] = 1
            if char == '\n':
                break
        return output

    model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,epochs=EPOCH,validation_split=0.2)

    model_train.save(model_name)
    encoder_infer.save(encoder_name)
    decoder_infer.save(decoder_name)

    for i in range(1,10):
        test = encoder_input[i:i+1,:,:]#i:i+1保持数组是三维
        out = predict_chinese(test,encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
        #print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
        print(input_texts[i])
        print(out)

if __name__ == '__main__':
        run(True)