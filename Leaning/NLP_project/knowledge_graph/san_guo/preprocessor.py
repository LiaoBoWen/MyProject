import pyltp
import os

ltp_path = '../../../Modules-studying/foolnltk/ltp_data_v3.4.0/'

def cut_words(sentences):
    segmentor = pyltp.Segmentor()
    segmentor.load(ltp_path + 'cws.model')
    words = [word for sentence in sentences for word in segmentor.segment(sentence)]
    segmentor.release()
    return words

def pos_words(words):
    postagger = pyltp.Postagger()
    postagger.load(ltp_path + 'pos.model')
    postags_lst = [pos for pos in postagger.postag(words)]
    postagger.release()
    return postags_lst



if __name__ == '__main__':
    words = cut_words(['你好,同志们好了,解散吧,哦,对了,廖博文,你留一下,我有点关于罗丹,王源,赵美丽的事要问你', '再见了,'])   # 会去除多余的空格
    postags = pos_words(words)
    target_pos = ('n', 'nh')
    result = []
    for i in range(len(postags)):
        if postags[i] in target_pos:
            result.append(words[i])
    print(result)