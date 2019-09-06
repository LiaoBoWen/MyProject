import os
import json
import datetime

import warnings
from collections import Counter

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings('ignore')


# 参数设置:

class TraingConfig:
    epoches = 10
    evaluateEvary = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig:
    embeddingSize = 200
    filters = 128       # todo
    numHeads = 8
    numBlocks = 1
    epsion = 1e-8       # todo 防止分母为零
    keepProp = 0.9      # todo

    dropoutKeepProb = 0.5   # todo
    l2RegLambda = 0.0

class Config:
    sequenceLength = 200  # todo
    batchSize = 128

    dataSource = '../data/preProcess/labeledTrain.csv'

    stopWrodSources = '../data/english'

    numClasses = 2

    rate = 0.8 # 训练集比例

    training = TraingConfig()

    model = ModelConfig()

config = Config()

# 数据预处理:

class Dataset:
    def __init__(self,config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequennceLength
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.bathSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.trainLables = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}


    def _readData(self,filePath):
        
        df = pd.read_csv(filePath)
        labels = df['sentiment'].tolist()
        review = df['review'].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels

    def _reviewProcess(self,review, sequenceLength, wordToIndex):
        '''单条数据进行id化'''
        reviewVec = np.zeros((sequenceLength))
        sequenceLength = sequenceLength

        if len(review) < sequenceLength:
            sequenceLength = len(review)

        for i in range(sequenceLength):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex['UNK']

        return reviewVec

    def _genTrainEvalData(self,x,y,rate):
        reviews = []
        labels = []

        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i],self._sequenceLength,self._wordToIndex)
            reviews.append(reviewVec)

            labels.append(y[i])

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex],dtype='int64')
        trainLabels = np.array(labels[:trainIndex],dtype='float32')

        evalReviews = np.asarray(reviews[trainIndex:],dtype='int64')
        evalLabels = np.array(labels[trainIndex:],dtype='float32')

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self,reviews):
        allWords = [word for review in reviews for word in review]

        subWords = [word for word in allWords if word not in self._stopWordDict]

        wordCount = Counter(subWords)

        sortedWordCount = wordCount.most_common()

        words = [item[0] for item in sortedWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        self._wordToIndex = dict(zip(vocab,range(len(vocab))))
        self._indexToWord = dict(zip(range(len(vocab)),vocab))

        with open('../data/wordJson/wordToIndex.json') as f:
            json.dump(self._wordToIndex,f)
        with open('../data/wordJson/indexToWord.json') as f:
            json.dump(self._indexToWord,f)

    def _getWordEmbedding(self,words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec")
        vocab = []
        wordEmbedding = []

        vocab.append('pad')
        vocab.append('UNK')
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.zeros(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + '不存在词向量中')

        return vocab, np.array(wordEmbedding)

    def _readStopWord(self,stopWordPath):
        with open(stopWordPath,'r') as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 使用字典的方式进行查找会快一点
            self.stopWordDict = dict(zip(stopWordList,range(len(stopWordPath))))

    def dataGen(self):
        '''初始化训练集和验证集'''
        self._readStopWord(self._stopWordSource)

        reviews, labels = self._readData(self._dataSource)

        self._genVocabulary(reviews)

        self.trainReviews, self.trainLabels, self.evalReviews, self.evalLabels = self._genTrainEvalData(reviews, labels, self._rate)


data = Dataset(config)
data.dataGen()

print('train data shape: {}'.format(data.trainReviews.shape))
print('train label sahpe: {}'.format(data.trainLabels.shape))
print('eval data shape: {}'.format(data.evalReviews.shape))


def nextBatch(x,y,batchSize):
    perm = np.arange(len(x))
    np.random.shuffle(perm)

    x = x[perm]
    y = y[perm]

    length = len(x)
    numBatches = (length - 1)// batchSize + 1

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start:min(end, length)],dtype='int64')
        batchY = np.array(y[start:min(end, length)],dtype='float32')

        yield batchX, batchY


# Positional Embedding
def fixedPositionEmbedding(batchSize,sequenceLen):

    embeddedPosition = []

    for _ in batchSize:
        embeddedPosition.append(np.eye(sequenceLen))

    return np.array(embeddedPosition, dtype='float32')


class Transformer:
    def __init__(self,config,wordEmbedding):
        self.inputX = tf.placeholder(tf.int32,[None.config.sequenceLength],name='inputX')
        self.inputY = tf.placeholder(tf.float32,[None,1],name='inputY')

        self.dropoutKeepProb = tf.placeholder(tf.float32,name='dropoutProb')
        self.embeddedPosition = tf.placeholder(tf.float32,[config.sequenceLength,config.sequenceLength], name='embeddedPosition')

        self.config = config

        l2Loss = tf.constant(0.0)

        # todo 词嵌入层, 位置向量的定义方式有两种：一是直接用固定的one-hot的形式传入，然后和词向量拼接，在当前的数据集上表现效果更好。
        #  另一种就是按照论文中的方法实现，这样的效果反而更差，可能是增大了模型的复杂度，在小数据集上表现不佳。

        with tf.name_scope('embedding'):
            # 直接使用的是word2vec训练的词向量而不是通过DL训练
            self.W = tf.Variable(tf.cast(wordEmbedding,dtype=tf.float32,name='word2Vec'),name='W')

            self.embedded = tf.nn.embedding_lookup(self.W,self.inputX)
            self.embeddedWords = tf.concat([self.embedded, self.embeddedPosition],-1)

        with tf.name_scope('transformer'):
            # for i in range(config.model.numBlocks):
            with tf.name_scope('transformer-{}'.format(1)):
                # todo 鸡肋！ 上一层的block完全没有传递到下一层来！
                multiHeadAtt = self._multiheadAttention(rawKeys=self.inputX,queries=self.embeddedPosition,
                                                        keys=self.embeddedWords)
                self.embeddedWords = self._feedForward(multiHeadAtt,
                                                       [config.model.filters,config.model.embeddingSize + config.sequenceLength])

            outputs = tf.reshape(self.embeddedWords,[-1,config.sequenceLength * (config.model.embeddingSize + config.sequenceLength)])

        outputSize = outputs.get_shape()[-1].value

        with tf.name_scope('dropout'):
            outputs = tf.nn.dropout(outputs,keep_prob=self.dropoutKeepProb)

        with tf.name_scope('output'):
            outputW = tf.get_variable(
                'outputW',shape=[outputSize,1],
                initializer=tf.contrib.layer.xavier_initializer()
            )
            outputB = tf.Variable(tf.constant(0.1,shape=[1],name='outputB'))
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(outputs,outputW,outputB)
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions,0.0),tf.float32,name='binaryPreds')

        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions,labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss


    def _layerNormalization(self,inputs,scop='layerNorm'):
        # layerNorm是在最后的维度上计算输入的数据的均值与方差,BN是考虑所有的维度
        # 最后的输出还要做一下 线性变化
        epsilon = self.config.model.epsilon

        inputShape = inputs.get_shape()   # [batch_size,sequence_len,embedding_len]

        paramsShape = inputShape[-1:]

        mean, variance = tf.nn.moments(inputs,[-1],keep_dims=True)  # TODO 注意keep_dim
        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))

        normalized = (input - mean) / (variance + epsilon) ** 0.5#  TODO 这里应该是标准差而不是方差

        outputs = gamma * normalized + beta

        return outputs


    def _multiheadAttention(self,rawKeys,queries,keys,numUnits=None,causality=False,scope='multiheadAttention'):
        numHeads = self.config.model.numHeads
        keepProb = self.config.model.keepProp

        if numUnits  is None:
            numUnits = queries.get_shape().as_list()[-1]


        # todo  在进行attention的时候,需要对数据进行映射,论文中是attention之后再映射,但是实际上是一样的,,激活函数时候的是relu
        Q = tf.layers.dense(queries,numUnits,activation=tf.nn.relu)
        K = tf.layers.dense(keys,numUnits,activation=tf.nn.relu)
        V = tf.layers.dense(keys,numUnits,activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q,numHeads,axis=-1),axis=0)
        K_ = tf.concat(tf.split(K,numHeads,axis=-1),axis=0)
        V_ = tf.concat(tf.split(V,numHeads,axis=-1),axis=0)

        similary = tf.matmul(Q_, tf.transpose(K_,[0,2,1]))

        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # todo 这是什么沙雕操作
        keyMasks = tf.tile(rawKeys,[numHeads,1])

        keyMasks = tf.tile(tf.expand_dims(keyMasks,1),[1,tf.shape(queries[1],1)])

        paddings = tf.ones_like(scaledSimilary) * (-2 ** 32 + 1)

        maskedSimily = tf.where(tf.equal(keyMasks,0),paddings,scaledSimilary)

        if causality:  # todo 这里是用于语言生成的时候使用,分类不需要
            diagVals = tf.ones_like(maskedSimily[0,:,:])
            tril = tf.contrib.linalg.LinearOperator(diagVals).to_dense()   # todo to_dense
            masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(maskedSimily)[0],1,1,])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            maskedSimily = tf.where(tf.equal(masks,0),paddings,maskedSimily)

        weights = tf.nn.softmax(maskedSimily)

        outputs = tf.matmul(weights,V_)

        outputs = tf.concat(tf.split(outputs,numHeads,axis=0),axis=2)

        outputs = tf.nn.dropout(outputs,keep_prob=keepProb)


        # todo 残差
        outputs += queries

        # TODO 注意在最后进行了归一化
        outputs = self._layerNormalization(outputs)

    def _feedForward(self,inputs,filters,scope='multiheadAttention'):
        '''使用卷积神经网络'''
        params = {'inputs':input,'filters':filters[0],'kernel_size':1,
                'activation':None,'use_bias':True}

        outputs = tf.layers.conv1d(**params)

        params = {'inputs':outputs,'filters':filters[1],'kernel_size':1,
                  'activation':None,'use_bias':True}

        outputs = tf.layers.conv1d(**params)

        outputs += inputs

        #todo 注意在这里也使用了归一化
        outputs = self._layerNormalization(outputs)

        return outputs

    def _positionEmbeddings(self,scope='positionEmbedding'):
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize

        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen),0),[batchSize,1])

        positionEmbedding = np.array([[pos / np.power(10000,(i - i % 2) / embeddingSize) for i in range(embeddingSize)] for pos in range(sequenceLen)])

        positionEmbedding[:,0::2] = np.sin(positionEmbedding[:,0::2])
        positionEmbedding[:,1::2] = np.cos(positionEmbedding[:,1::2])

        positionEmbedding_ = tf.cast(positionEmbedding,dtype=tf.float32)

        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_,positionIndex)

        return positionEmbedded


def mean(item):
    return sum(item) / len(item)

def genMetrics(trueY,predY,binaryPredY):
    auc = roc_auc_score(trueY,predY)
    accuracy = accuracy_score(trueY,binaryPredY)
    precision = precision_score(trueY,binaryPredY)
    recall = recall_score(trueY,binaryPredY)

    return round(accuracy,4),round(auc,4),round(precision,4),round(recall,4)



# Start
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
embeddedPosition = fixedPositionEmbedding(config.batchSize,config.sequenceLength)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,log__device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=session_conf) as sess:
        transformer = Transformer(config,wordEmbedding)

        globalStep = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        gradsAndVars = optimizer.compute_gradients(transformer.loss)
        trainOp = optimizer.apply_gradients(gradsAndVars,global_step=globalStep)


        # 梯度可视化
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram('{}/grad/hist'.format(v.name),g)
                tf.summary.scalar('{}/grad/sparsity'.format(v.name),tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir,'summarys'))
        print('Writting to {}\n'.format(outDir))

        lossSummary = tf.summary.scalar('loss',transformer.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir,'train')
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir,sess.graph)

        evalSummaryDir = os.path.join(outDir,'eval')
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir,sess.graph)

        saver = tf.train.Saver(tf.global_variables,max_to_keep=5)

        sess.run(tf.global_variables_initializer)

        def trainStep(batchX,batchY):
            feed_dict = {
                transformer.inputX:batchX,
                transformer.inputY:batchY,
                transformer.dropoutKeepProb:config.model.dropoutKeepProb,
                transformer.embeddedPosition:embeddedPosition
            }

            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp,summaryOp,globalStep,transformer.loss,transformer.predictions,transformer.binaryPreds],
                feed_dict
            )
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, auc:{}, acc:{}, precision:{}, recall:{}".format(timeStr, step, loss, acc, auc, precision, recall))
            trainSummaryWriter.add_summary(summary,step)

        def devStep(batchX,batchY):
            feed_dict = {
                transformer.inputX:batchX,
                transformer.inputY:batchY,
                transformer.dropoutKeepProb:1.0,
                transformer.embeddedPosition:embeddedPosition
            }

            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp,globalStep, transformer.loss, transformer.predictions, transformer.binaryPreds],
                feed_dict = feed_dict
            )

            acc, auc, predcision, recall = genMetrics(batchY,predictions,binaryPreds)

            evalSummaryWriter.add_summary(summary,step)

            return loss, auc, auc, precision_score, recall

        for i in range(config.training.epoches):
            print('Start training model.')
            for batchTrain in nextBatch(trainReviews,trainLabels,config.batchSize):
                trainStep(batchTrain[0],batchTrain[1])

                if globalStep % config.training.evaluateEvary == 0:
                    print('\n Evaluation: ')
                    losses = []
                    accs = []
                    aucs = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews,evalLabels,config.batchSize):
                        loss, acc, auc, precision, recall = devStep(batchEval[0],batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    print('{}, step:{}, loss:{}, acc:{}, auc:{}, precision:{}. recall:{} '.format(time_str,globalStep,mean(losses),mean(accs),mean(aucs),mean(precisions),mean(recalls)))
