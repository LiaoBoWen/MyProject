import logging
import os
import codecs
import sys
import re

def getCotent(filename):
    with open(filename,'r') as f:
        content = f.readlines()
    return content

def push_alll():
    classes = ['neg', 'pos']
    for class_ in classes:
        # inp = './data/{}'.format(class_)
        path_ = './data/{}_all.txt'.format(class_)
        if not os.path.exists(path_):
            with open(path_, 'a') as writer:
                neg_all = list(map(lambda x: os.path.join('./data/{}/'.format(class_) + x, ),
                                   os.listdir('./data/{}'.format(class_))))
                print(neg_all)
                for file_path in neg_all:
                    try:
                        with open(file_path) as n_tmp:
                            content = n_tmp.read()
                            content = re.sub(r'\n', '', content)
                            writer.write(content + '\n')
                    except:
                        pass
            logger.info('Saved {} file'.format(class_))

def all_data():
    with open('./data/all.txt','w',encoding='utf8') as F:
        with open('./data/pos_all.txt') as f:
            content1 = f.read()

        with open('./data/neg_all.txt') as f:
            content2 = f.read()
        F.write(content1 + content2)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)

    all_data()