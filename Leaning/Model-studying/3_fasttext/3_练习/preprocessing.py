import re
import jieba
import random

def preprocession(path='./toutiao_cat_data.txt'):
    new_vocab = []

    with open(path,'r',encoding='utf8') as f:
        test = f.read()
        re_tip = re.findall(r'[0-9]{19}_!_[0-9]{3}_!_(.+?)_!_(.+)',test)
        for label,vocab  in re_tip:
            raw_cut_word = ''.join(vocab.strip().strip('_!_').split('_!_'))
            cut_word = ' '.join(list(jieba.cut(raw_cut_word)))
            new_vocab.append(cut_word + '\t__label__' + label+ '\n')
        new_vocab.sort(key=lambda x: random.random())
        with open('cleaned_data.txt','w',encoding='utf8') as f1:
            f1.writelines(new_vocab)


def train_test_split(train_scale=0.6):
    with open('cleaned_data.txt','r',encoding='utf8' )as f1:
        content = f1.readlines()
        content_len = len(content)
        train_size = int(content_len * train_scale)
        with open('train.txt','w',encoding='utf8') as f2:
            f2.writelines(content[:train_size])
        with open('test.txt','w',encoding='utf8') as f3:
            f3.writelines(content[train_size:])


if __name__ == '__main__':
    preprocession()
    train_test_split()