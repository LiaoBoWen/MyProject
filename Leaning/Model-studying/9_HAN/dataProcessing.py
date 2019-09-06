import tensorflow as tf
import os
import csv
import re
import itertools
import more_itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
# from bs4 import BeautifulSoup
from utils import prjPaths
import pickle

class IMDB:
    def __init__(self,action):
        '''预处理imdb数据'''
        self.paths = prjPaths()
        self.ROOT_DATA_DIR = self.ROOT_DATA_DIR
        self.DATASET = 'imdb'

        self.CSVFILENAME = os.path.join(self.ROOT_DATA_DIR,self.DATASET,'{}.csv'.format(self.DATASET))
        assert action in ['create','fetch'], 'invalid action'

        if action == 'create':
            if os.path.exists(self.CSVFILENAME):
                print('removing existing csv file from {}'.format(self.CSVFILENAME))
                os.remove(self.CSVFILENAME)

            train_dir = os.path.join(self.ROOT_DATA_DIR,self.DATASET,'acImdb','train')
            test_dir = os.path.join(self.ROOT_DATA_DIR,self.DATASET,'acImdb','test')

            trainPos_dir = os.path.join(train_dir,'pos')
            trainNeg_dir = os.path.join(train_dir,'neg')

            testPos_dir = os.path.join(test_dir,'pos')
            testNeg_dir = os.path.join(test_dir,'neg')

            self.data = {'trainPos':self._getDirCountents(trainPos_dir),
                         'trainNeg':self._getDirCountents(trainNeg_dir),
                         'testPos':self._getDirCountents(testPos_dir),
                         'testNeg':self._getDirCountents(testNeg_dir)}

    def _getDirCountents(self,path):
        dirFiles = os.listdir(path)
        dirFiles = [os.path.join(path,file) for file in dirFiles]
        return dirFiles

    def _getID_label(self,file,binary):
        splitFile = file.split('/')
        testOtrain = splitFile[-3]
        filename = os.path.splitext(splitFile[-1])[0]  # todo 这个？
        id, label = filename.split('_')
        if binary:
            if int(label) < 5:
                label = 0
            else:
                label = 1
        return [id, label, testOtrain]

    def _loadTextFiles(self,dirFiles,binary):
        TxtContents = []
        for file in tqdm(dirFiles,desc='process all file in a directory'):
            try:
                with open(file,encoding='utf8') as txtFile:
                    content = txtFile.read()
                    id, label, testOtrain = self._getID_label(file,binary=binary)
                    TxtContents.append({'id':id,'content':content,'label':label,'testOtrain':testOtrain})
            except:
                print(f'"{file}" file threw and error and is being omited')
                continue
        return TxtContents

    def _writeTxtFiles(self,TxtContents):
        with open(self.CSVFILENAME,'a') as csvFile:
            fileNames = ['id','content','label','testOtrain']
            writer =  csv.DictWriter(csvFile,filenames=fileNames)
            writer.writeheader()

            for seq in TxtContents:
                try:
                    writer.writerow({'id':seq['id'],
                                     'content':seq['content'].encode('ascii','ignore').decode('ascii'),
                                     'label':seq['label'],
                                     'testOtrain':seq['testOtrain']})
                except:
                    print(f'this sequence threw an exception :{seq["id"]}')
                    continue

    def createMaanger(self,binary):
        for key in self.data.keys():
            self.data[key] = self._loadTextFiles(self.data[key],binary)
            self._writeTxtFiles(self.data[key])


    def _clean_str(self,string):
        # 英文文本处理
        string = re.sub(r'[^A-za-z0-9(),!?\"\`]',' ',string)
        string = re.sub(r'\'s',' \'s',string)
        string = re.sub(r'\'ve',' \'ve',string)
        string = re.sub(r'n\'t',' n\'t',string)
        string = re.sub(r'\'re',' \'re',string)
        string = re.sub(r'\'d',' \'d',string)
        string = re.sub(r'\'ll',' \'ll',string)
        string = re.sub(r',',' , ',string)
        string = re.sub(r'.',' . ',string)
        string = re.sub(r'!',' ! ',string)
        string = re.sub(r'\(',' \( ',string)
        string = re.sub(r'\)',' \) ',string)
        string = re.sub(r'\?',' \? ',string)
        string = re.sub(r'\s{2,}',' ',string)
        return string

    def _oneHot(self,ys):
        '''
        :param ys: dataset labels
        :return:
        '''
        y_train, y_test = ys
        y_train = list(map(int,y_train))
        y_test = list(map(int,y_test))
        lookuplabels = {v:k for k,v in enumerate(sorted(list(set(y_train + y_test))))}
        record_y_train = [lookuplabels[i] for i in y_train]
        record_y_test = [lookuplabels[i] for i in y_test]
        labels_y_train = tf.constant(record_y_train)
        labels_y_test = tf.constant(record_y_test)
        max_label = tf.reduce_max(labels_y_train + labels_y_test)
        labels_y_train_OHE = tf.one_hot(labels_y_train,max_label + 1)
        labels_y_test_OHE = tf.one_hot(labels_y_test,max_label + 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y_train_ohe = sess.run(labels_y_train_OHE)
            y_test_ohe = sess.run(labels_y_test_OHE)
        return [y_train_ohe,y_test_ohe,lookuplabels]

    def _index(self,xs):
        '''
        :param xs: text data
        :return:
        '''
        def _apply_index(txt_data):
            indexed = [[[unqVoc_LookUp[char] for char in seq] for seq in doc] for doc in txt_data]
            return indexed

        x_train, x_test = xs

        unqVoc = set(list(more_itertools.collapse(x_train[:])))
        unqVoc_LookUp = {k: v+1 for v,k in enumerate(unqVoc)}
        vocab_size = len(unqVoc_LookUp)

        x_train = _apply_index(txt_data=x_train)
        x_test = _apply_index(txt_data=x_test)

        max_seq_len = max([len(sent) for sent in (x_train + x_test)])
        max_sent_len = max([len(sent) for sent in (x_train + x_test)])

        persisted_vars = {'max_seq_len':max_seq_len,
                          'max_sent_len':max_sent_len,
                          'vocab_size':vocab_size}
        return [x_train, x_test, unqVoc_LookUp, persisted_vars]

    def partitionManager(self,dataset):
        df = pd.read_csv(self.CSVFILENAME)
        train = df.loc[df['testOtrain'] == 'train']
        test = df.loc[df['testOtrain'] == 'test']

        create3DList = lambda df : [[self._clean_str(seq) for seq in '|||'.join(re.split('[.?!]',docs)).split('|||')]
                                    for docs in df['content'].values]
        x_train = create3DList(df=train)
        x_test = create3DList(df=test)

        x_train, x_test, unqVoc_Lookup, persisted_vars = self._index(xs=[x_train[:],x_test[:]])

        y_train = train['label'].tolist()
        y_test = test['label'].tolist()

        y_train_ohe, y_test_ohe, lookuplabels = self._oneHot(ys=[y_train,y_test])

        persisted_vars['lookuplabels'] = lookuplabels

        persisted_vars['num_classes'] = len(lookuplabels.keys())

        if not os.path.exists(os.path.join(self.paths.LIB_DIR,self.DATASET)):
            os.mkdir(os.path.join(self.paths.LIB_DIR,self.DATASET))
        pickle._dump(unqVoc_Lookup,open(os.path.join(self.paths.LIB_DIR,self.DATASET,'unqVoc_Lookup.p'),'wb'))
        pickle._dump(unqVoc_Lookup,open(os.path.join(self.paths.LIB_DIR,self.DATASET,'persisted_vars.p'),'wb'))

        return [x_train, y_train_ohe,x_test,y_test_ohe]

    def get_data(self,type_):
        print(f'loading {type_} dataset ...')

        x = np.load(os.path.join(self.paths.ROOT_DATA_DIR,self.DATASET,f'{type_}_x.npy'))
        y = np.load(os.path.join(self.paths.ROOT_DATA_DIR,self.DATASET,f'{type_}_y.npy'))

        docsize = np.load(os.path.join(self.paths.ROOT_DATA_DIR,self.DATASET,f'{type_}_docsize.npy'))
        sent_size = np.load(os.path.join(self.paths.ROOT_DATA_DIR,self.DATASET,f"{type_}_sent_size.npy"))

        return [x, y, docsize, sent_size]

    def get_batch_iter(self,data,batch_size,num_epochs,shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) -1) / batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                next_batch = data[shuffle_indices]
            else:
                next_batch = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size,data_size)
                yield epoch, next_batch[start_index:end_index]

    def hanformater(self,inputs):
        batch_size = len(inputs)
        document_sizes = np.array([len(doc) for doc in inputs],dtype=np.int32)
        document_size = document_sizes.max()

        sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
        sentence_size = max(map(max,sentence_sizes_))

        b = np.zeros(shape=[batch_size,document_size,sentence_size],dtype=np.int32)

        sentence_sizes = np.zeros(shape=[batch_size,document_size],dtype=np.int32)
        for i, document in enumerate(tqdm(inputs,desc='THIS IS HAN')):
            for j,sentence in enumerate(document):
                sentence_size[i,j] = sentence_sizes_[i][j]
                for k, word in enumerate(sentence):
                    b[i, j, k] = word
        return b, document_sizes, sentence_sizes
