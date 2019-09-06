import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn

# 过滤函数
def clean_str(string):
    '''

    :param string:
    :return:
    '''
    string = re.sub(r'[^A-Za-z0-9(),!?\'\`]',' ',string)
    string = re.sub(r"\'s", " \'s",string)
    string = re.sub(r"\'ve", " \'ve",string)
    string = re.sub(r"n\'t", " n\'t",string)
    string = re.sub(r"\'re", " \'re",string)
    string = re.sub(r"\'d", " \'d",string)
    string = re.sub(r"\'ll", " \'ll",string)
    string = re.sub(r",", " , ",string)
    string = re.sub(r"!", " ! ",string)
    string = re.sub(r"\(", " ( ",string)
    string = re.sub(r"\)", " ) ",string)
    string = re.sub(r"\?", " ? ",string)
    string = re.sub(r"\s{2,}", " ",string)
    return string.strip().lower()

# 加载数据
def load_data_and_labels(positive_data_file,negative_data_file):
    '''
    这里的数据是先进行分条、去除两端的空格、换行符，对形成的数据列表进行数据的清洗
    加载数据，注意编码不是utf-8
    :param positive_data_file:
    :param negative_data_file:
    :return:英文分词 、 labels
    '''
    with open(positive_data_file,encoding='ISO-8859-1') as pos:
        with open(negative_data_file,encoding='ISO-8859-1') as neg:
            # 去除换行符
            positive_examples = pos.read().split('\n')[:-1]
            negative_examples = neg.read().split('\n')[:-1]


            # 去除开始或者结尾的空格
            positive_examples = [s.strip() for s in positive_examples]
            negative_examples = [s.strip() for s in negative_examples]

            # 合并两类数据
            x_text = positive_examples + negative_examples
            x_text = [clean_str(sent) for sent in x_text]

            # 生成labels
            positive_labels = [[0,1] for _ in positive_examples]
            negative_labels = [[1,0] for _ in negative_examples]

            y = np.concatenate([positive_labels,negative_labels],0)
            return [x_text, y]


def load_data_labels(data_file,labels_file):
    data = []
    labels = []

    with open(data_file,encoding='latin-1') as f:
        data.extend([s.strip() for s in f.readlines()])
        data = [clean_str(s) for s in data]

    with open(labels_file) as f:
        labels.extend([s.strip() for s in f.readlines()])
        labels = [label.split(',') for label in labels]

    lb = LabelBinarizer()       # todo ?啥东西
    y = lb.fit_transform(labels)

    vocab_processor = learn.preprocessing.VocabularyProcessor(1000) # todo 干嘛的？
    x = np.array(list(vocab_processor.fit_transform(data)))

    return x, y, vocab_processor

# 生成batch数据的生成器
def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1   # 常用batch操作~

    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(data_size)
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]


if __name__ == '__main__':
    # 测试
    load_data_and_labels('./data/rt-polarity.pos','./data/rt-polarity.neg')