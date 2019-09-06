import os
import fastText.FastText as ft
from sklearn.metrics import classification_report, confusion_matrix

'''
# 这里的分类的效果并没有达到很好，有一类由于数据量过小（125条），所以准确率太低
# 这里的语料没有进行很好的预处理，没有去除停用词，由于还是短文本，含有15个种类
# 不知道时候还有其他的预处理文本数据的方法

=============================
# 初步的数据清洗
=============================
def go_split(s,min_len):
    # 拼接正则表达式
    symbol = '，；。！、？!'
    symbol = "[" + symbol + "]+"
    # 一次性分割字符串
    result = re.split(symbol, s)
    return [x for x in result if len(x)>min_len]

def is_dup(s,min_len):
    result = go_split(s,min_len)
    return len(result) !=len(set(result))

def is_neg_symbol(uchar):
    neg_symbol=['!', '0', ';', '?', '、', '。', '，']
    return uchar in neg_symbol
'''


def read_file(predictPath='test.txt'):
    content_list, labels = [], []
    with open(predictPath,encoding='utf8') as f:
        for line in f:
            try:
                content, label = line.strip().split('\t')
                if content:
                    content_list.append(content)
                    labels.append(label)
            except:
                pass
    return content_list, labels

def train_fasttext(inputPath='train.txt',savePath='./model.m',label='__label__'):
    if not os.path.exists(savePath):
        print('train...')
        classfication = ft.train_supervised(inputPath,label=label)
        classfication.save_model(savePath)
    else:
        classfication = ft.load_model(savePath)

    print('load model...')

    return classfication

def run_train(model,predictPath='test.txt'):
    print('predict...')
    # test = model.test(testPath)
    contents, labels = read_file(predictPath)

    # 不包含label的分词后的content
    # todo predict可以直接预测文本也可以预测分词后的文本
    predict_result = model.predict(contents)

    # 这里由于是windows所以predict的结果这么输出
    predict_label = predict_result[0]

    # 输出分类后的效果
    report_result = classification_report(labels,predict_label)
    print(report_result)

    # 输出混淆矩阵
    cm = confusion_matrix(labels,predict_label)
    print(cm)


if __name__ == '__main__':
    model= train_fasttext()
    run_train(model)