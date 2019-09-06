import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(all_predictions,all_labels):
    '''有时候0.5或许不是最佳的分界线，所以把0~1分成一百份，
    分别计算依次为边界的f1-score，从里面找到最佳的分界线'''
    all_predictions = np.ravel(all_predictions)
    all_labels = np.ravel(all_labels)


    thresholds = np.linspace(0.1,1,100)
    all_f1s = []

    for threshold in thresholds:
        preds = (all_predictions >= threshold).astype('int')
        f1 = f1_score(all_labels,preds)
        all_f1s.append(f1)

    best_threshold = thresholds[int(np.argmax(np.array(all_f1s)))]
    print('best threshold is {}'.format(best_threshold))
    print(all_f1s)

    return best_threshold