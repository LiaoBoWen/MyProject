# export model

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


with open('text_classifier_output_file/test_results.tsv',encoding='utf8') as f:
    predict_data = pd.read_csv(f,sep='\t',header=None)
    predict_data.values.argmax(1) - 1
    print(predict_data.values)
    predict_data = predict_data.apply(lambda x: np.argmax(x) - 1, axis=1)

with open('data/test.tsv',encoding='utf8') as f:
    test_data = pd.read_csv(f,sep='\t')
    labels = test_data['label']

print(accuracy_score(labels,predict_data))
