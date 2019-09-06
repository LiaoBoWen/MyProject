import numpy as np
import re

def textParse(text):
    text = text.lower()
    regEx = re.compile(r'[a-z]|\d')
    words = regEx.split(text)
    words = [word for word in words if len(word) >0]
    return words

def loadSMSData(path):
    print('filename:',path)

    classCategory = []
    smsWords = []

    with open(path) as f:
        for line in f.readlines():
            line_words = line.strip().split('\t')
            if line_words[0] == 'ham':
                classCategory.append(0)
            else:
                classCategory.append(1)
            words = textParse(line_words[1]) # 文本内容进行解析
            smsWords.append(words)
        return smsWords, classCategory

def createVocab(smsWords):
    '''建立词表'''
    vocabSet = set([])
    for words in smsWords:
        # 并
        vocabSet |= set(words)
    vocab = list(vocabSet)
    return vocab

def get_vocab(path):
    ''' ???? '''
    with open(path,encoding='utf8') as f:
        vocab = f.readline().strip().splilt('\t')
    return vocab

def setOfWord2Vec(vocabList,smsWords):
    '''统计单词出现频率'''
    vocabMarked = [0] * len(vocabList)
    for smsWord in smsWords:
        if smsWord in vocabList:
            vocabMarked[vocabList.index(smsWord)] += 1
    return vocabMarked

def setOfVec2Mat(vocabList,smsWordsList):
    '''构建向量矩阵'''
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWord2Vec(vocabList,smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList

def trainNaiveBayes(trainMarkedWords,trainCategory):
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])

    pSpam = sum(trainCategory) / float(numTrainDoc)

    # 统计每个词在不同的类别出现的频率 ，这里为了进行了平滑操作，分子初始为1，分母初始为0
    wordsInSpanNum = np.ones(numWords)
    wordsInHealthNum = np.ones(numWords)
    spamWordsNum = 2
    healthWordsNum = 2

    for i in range(0,numTrainDoc):
        # 如果是垃圾邮件
        if trainCategory[i] == 1:
            wordsInSpanNum += trainMarkedWords[i]
            spamWordsNum += sum(trainMarkedWords[i])
        else:
            wordsInHealthNum += trainMarkedWords[i]
            healthWordsNum += sum(trainMarkedWords[i])

    pWordsSpanicity = np.log(wordsInSpanNum / spamWordsNum)
    pWordsHealthy = np.log(wordsInHealthNum / healthWordsNum)

    return pWordsSpanicity, pWordsHealthy

def getTrainModelInfo():
    vocabList = get_vocab('./data/vacabularyList.txt')
    pWordsHealthy = np.loadtxt('./data/pWordsHealthy.txt',delimiter='\t')
    pWordsSpamcity = np.loadtxt('./data/pWordsSpamcity.txt',delimiter='\t')
    with open('./data/pSam.txt') as f:
        pSpam = float(f.readline().strip())
    return vocabList, pWordsSpamcity, pWordsHealthy, pSpam

def classify(vocabList,pWordsSpam,pWordsHealth,pSpam,testWords):
    testWordsCount = setOfWord2Vec(vocabList,testWords)
    testWordsMarkedArray = np.array(testWordsCount)

    p1 = sum(testWordsMarkedArray * pWordsSpam) + np.log(pSpam)
    p0 = sum(testWordsMarkedArray * pWordsHealth) + np.log(1 - pSpam)

    if p1 > p0:
        return 1
    else:
        return 0