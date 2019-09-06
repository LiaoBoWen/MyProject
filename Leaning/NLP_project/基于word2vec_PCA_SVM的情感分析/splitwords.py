import jieba

def cutWords(sourceFile,targetFile):
    stopWords = readStop()
    with open(targetFile,'a',encoding='utf8') as t:
        with open(sourceFile,'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = jieba.cut(line)
                # words = [w for w in line if w not in stopWords]
                line = ' '.join(line)
                t.write(line + '\n')

def readStop(file='./data/stopWord.txt'):
    with open(file,encoding='utf8') as f:
        content = f.read().split('\n')
        return content



if  __name__ == '__main__':
    cutWords('./data/neg_all.txt','./data/neg_split.txt')
    cutWords('./data/pos_all.txt','./data/pos_split.txt')