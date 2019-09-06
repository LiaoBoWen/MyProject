from keras.models import load_model


from config import CONFIG

import numpy as np
import pickle
import re

NUM_SAMPLES = 10000

def data_util(english_to_chinese=True):
    if english_to_chinese:
        INPUT_LENGTH = CONFIG['EN_len']
        OUTPUT_LENGTH = CONFIG['CH_len']
        INPUT_FEATURE_LENGTH = CONFIG['EN_feature_len']
        OUTPUT_FEATURE_LENGTH = CONFIG['CH_feature_len']
    else:
        INPUT_LENGTH = CONFIG['CH_len']
        OUTPUT_LENGTH = CONFIG['EN_len']
        INPUT_FEATURE_LENGTH = CONFIG['CH_feature_len']
        OUTPUT_FEATURE_LENGTH = CONFIG['EN_feature_len']

    input_dict = pickle.load(open('./model/input_dict{}.pkl'.format(english_to_chinese),'rb'))
    target_dict = pickle.load(open('./model/target_dict{}.pkl'.format(english_to_chinese),'rb'))
    target_dict_reverse = pickle.load(open('./model/target_dict_reverse{}.pkl'.format(english_to_chinese),'rb'))

    return  INPUT_LENGTH, OUTPUT_LENGTH, INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, input_dict, target_dict, target_dict_reverse

# 英语翻译为中文
en2zh_INPUT_LENGTH, en2zh_OUTPUT_LENGTH, en2zh_INPUT_FEATURE_LENGTH, en2zh_OUTPUT_FEATURE_LENGTH, en2zh_input_dict, en2zh_target_dict, en2zh_target_dict_reverse = data_util(english_to_chinese=True)
en2zh_model_train = load_model('./model/en2zh_model.h5')
en2zh_encoder_infer = load_model('./model/en2zh_encoder.h5')
en2zh_decoder_infer = load_model('./model/en2zh_decoder.h5')

# 中文翻译为英语
zh2en_INPUT_LENGTH, zh2en_OUTPUT_LENGTH, zh2en_INPUT_FEATURE_LENGTH, zh2en_OUTPUT_FEATURE_LENGTH, zh2en_input_dict, zh2en_target_dict, zh2en_target_dict_reverse = data_util(english_to_chinese=False)
zh2en_model_train = load_model('./model/zh2en_model.h5')
zh2en_encoder_infer = load_model('./model/zh2en_encoder.h5')
zh2en_decoder_infer = load_model('./model/zh2en_decoder.h5')

def predict(source,encoder_inference,decoder_inference,n_steps,features,target_dict,target_dict_reverse):
    state = encoder_inference.predict(source)
    predict_seq = np.zeros([1,1,features])
    predict_seq[0,0,target_dict['\t']] = 1

    output = ''
    for i in range(n_steps):
        yhat, h, c = decoder_inference.predict([predict_seq] + state)
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':
            break
    return output

while True:
    tip = '请输入[English/中文]: '
    input_str = input(tip)
    if input_str is None or input_str.strip() == "":
        continue
    if input_str == r'\b':
        print('再见！')
        exit()

    input_str = input_str.strip().lower()

    if bool(re.search(r'[a-zA-Z]',input_str)):
        if len(input_str) > en2zh_INPUT_LENGTH:
            print('输入太长，请重新输入')
            continue
        test = np.zeros((1,en2zh_INPUT_LENGTH,en2zh_INPUT_FEATURE_LENGTH))
        for char_index, char in enumerate(input_str):
            test[0,char_index,en2zh_input_dict[char]] = 1
        out = predict(test,en2zh_encoder_infer,en2zh_decoder_infer,en2zh_INPUT_LENGTH,en2zh_OUTPUT_FEATURE_LENGTH,en2zh_target_dict,en2zh_target_dict_reverse)
    else:
        if len(input_str) > zh2en_INPUT_LENGTH:
            print('输入太长， 请重新输入')
            continue
        test = np.zeros([1,zh2en_INPUT_LENGTH,zh2en_INPUT_FEATURE_LENGTH])
        for char_index, char in enumerate(input_str):
            test[0,char_index,zh2en_input_dict[char]] = 1
        out = predict(test,zh2en_encoder_infer,zh2en_decoder_infer,zh2en_OUTPUT_LENGTH,zh2en_OUTPUT_FEATURE_LENGTH,zh2en_target_dict,zh2en_target_dict_reverse)
    print(out)