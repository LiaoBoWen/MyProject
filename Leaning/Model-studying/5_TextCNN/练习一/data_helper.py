import numpy as np
import re
import jieba
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn

# 清洗中文文本，去除标点符号数字以及特殊符号
def clean_text(content):
    text = re.sub(r'[!@#$%^&*()_+`=-\\><?":{}|]+','',content)
    text = re.sub(r'[~！#￥%……&*（）]]','',text)
    text = re.sub(r'\d+','',text)
    text = re.sub(r'\s+','',text)
    return text

# 中文分词
def seg_text(text,stopWord_path='stopword.txt'):
    stop = [line.strip() for line in open(stopWord_path,encoding='utf8').readlines()]
    text_seged = jieba.cut(text.strip())
    output = [w for w in text_seged if w not in stop]
    return ' '.join(output)

# 过滤函数
def clear_str(string):
    string = re.sub(r'[^A-Za-z0-9(),?!\`\']',' ',string)
    string = re.sub(r'\'s',' \'s',string)
    string = re.sub(r'\'ve',' \'ve',string)
    string = re.sub(r'n\'t',' n\'t',string)
    string = re.sub(r'\'re',' \'re',string)
    string = re.sub(r'\'d',' \'d',string)
    string = re.sub(r' \'ll',' \'ll',string)
    string = re.sub(r',',' , ',string)
    string = re.sub(r'!',' ! ',string)
    string = re.sub(r'\(',' \( ',string)
    string = re.sub(r'\)',' \) ',string)
    string = re.sub(r'\?',' ? ',string)
    string = re.sub(r'\s{2,}',' ',string)

    return string.strip().lower()

def load_data_and_labels(positive_data_file,negative_data_file):
    # 这里的数据是先进行分条、去除两端的空格、换行符，对形成的数据列表进行数据的清洗
    with open(positive_data_file,encoding='ISO-8859-1') as pos:
        with open(negative_data_file,encoding='ISO-8859-1') as neg:
            # 去除换行符
            positive_examples = pos.read().split('\n')[:-1]
            negative_examples = neg.read().split('\n')[:-1]

    # 去除开始或者结尾的空格
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clear_str(sent) for sent in x_text]

    positive_labels = [[1,0] for _ in range(len(positive_examples))]
    negative_labels = [[0,1] for _ in range(len(negative_examples))]

    y = np.concatenate([positive_labels,negative_labels],0)

    return [x_text,y]

def batch_iter(data,batch_size,num_epochs,shuffle=True):
    if isinstance(data,list):
        data = np.array(data)
    data_size = data.shape[0]
    num_batch_per_epoch = int((data_size - 1) / batch_size) + 1 # 常用的batch操作

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(data_size)
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]

if __name__ == '__main__':
    data, y = load_data_and_labels('../data/rt-polarity.pos','../data/rt-polarity.neg')
    print(data)