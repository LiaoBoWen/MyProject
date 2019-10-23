from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import jieba

class Classifier:
    def __init__(self):
        self.get_train_data()
        self.train()


    def get_train_data(self):
        self.X, self.y = [], []
        with open('./question/label.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                y_, question = line.split('    ')
                question = question.strip()
                words = ' '.join([word for word in jieba.cut(question)])
                self.X.append(words)
                self.y.append(y_)


    def train(self):
        # 这里会过滤单个字的word,但是我们通过模型进行的类别的判定,所以对于多个字的word更加关注,所以,单个字过滤会有更好的效果
        self.tfidf = TfidfVectorizer()
        self.X = self.tfidf.fit_transform(self.X).toarray()
        # 加0.01进行平滑
        self.classifier = MultinomialNB(alpha=0.1)
        self.classifier.fit(self.X, self.y)
        predict_y = self.classifier.predict(self.X)
        # print('accuracy in train:{}'.format(accuracy_score(predict_y, self.y)))


    def predict(self, question):
        question = [' '.join([word for word in jieba.cut(question)])]
        question = self.tfidf.transform(question).toarray()
        y_predict = self.classifier.predict(question)[0]
        return int(y_predict)


if __name__ == '__main__':
    test = Classifier()
    result = test.predict('攀登者的评分特别高，那它的演员是谁呀')
    print(result)