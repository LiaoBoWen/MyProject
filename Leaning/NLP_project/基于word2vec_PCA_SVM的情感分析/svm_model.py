from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def read_data(data_path='./Data.csv'):
    Data = pd.read_csv(data_path,header=None)
    Y = Data.iloc[:,-1]
    X = Data.iloc[:,:-1]
    return X, Y


def train(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=32)

    pca = PCA(n_components=3,random_state=32)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    # x_train_pca = x_train
    # x_test_pca = x_test

    # clf = SVC(C=2.5,probability=True)
    # clf.fit(x_train_pca,y_train)
    clf = LogisticRegression(C=1,random_state=32)
    clf.fit(x_train_pca,y_train)
    print(metrics.accuracy_score(y_train,clf.predict(x_train_pca)))
    print(metrics.accuracy_score(y_test,clf.predict(x_test_pca)))

if __name__ == "__main__":
    X, Y = read_data()
    train(X,Y)