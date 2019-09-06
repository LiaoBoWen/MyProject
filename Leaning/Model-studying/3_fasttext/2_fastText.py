import logging
import os
import fastText.FastText as ff
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

# todo win和linux的fasttext的操作有点不太一样！
# https://blog.csdn.net/meyh0x5vDTk48P2/article/details/79055553
# ===============================================
'''
linux下的调用方式     
import fasttext
#训练模型
classifier = fasttext.supervised("data/try_fasttext_train.txt","data/try_fasttext.model",label_prefix="__label__")
 
#load训练好的模型
#classifier = fasttext.load_model('data/try_fasttext.model.bin', label_prefix='__label__')
 
result = classifier.test("data/try_fasttext_test.txt")
print(result.precision)

'''
# ===============================================


def train_fasttext_win(inputPath='news_fasttext/news_fasttext_train.txt',savePath='model.m',label='__label__'):
    if not os.path.exists(savePath):
        print('train model...')
        classifier = ff.train_supervised(inputPath,label=label)
        classifier.save_model(savePath)    # 保存模型
    else:
        classifier = ff.load_model('model.m')  # 读取模型
    print('loaded model...')
    return classifier


def win10_way(model):
    print('test...')
    test = model.test('news_fasttext/news_fasttext_test.txt')
    print(test)
    predict = model.predict(['你好吗','你好呀'])    # todo ？输入是个啥？
    print('predict: ',predict)


def main(machine = '__win__'):
    classifier = train_fasttext_win()
    win10_way(classifier)

main()