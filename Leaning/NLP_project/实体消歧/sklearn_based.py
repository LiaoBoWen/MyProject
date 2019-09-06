import re
import jieba
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



entity_data = pd.read_csv('./data/entity_disambiguation/entity_list.csv',encoding='utf8')
valid_data = pd.read_csv('./data/entity_disambiguation/valid_data.csv',encoding='gb2312')

print(entity_data.head(3))

s = ''
keywords_list = []
for i in entity_data['entity_name'].values.tolist():
    s += i + '|'
for k, v in Counter(s.split('|')).items():
    if v > 1:
        keywords_list.append(k)


train_sentences = []
for sentence in entity_data['desc']:
    train_sentences.append(' '.join(jieba.cut(sentence)))

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(train_sentences)

def get_entityid(sentence):
    id_start = 1001
    a = [' '.join(jieba.cut(sentence))]
    sim = cosine_similarity(tfidf.transform(a),X)[0]
    top_idx = np.argsort(sim)[-1]
    return id_start + top_idx

result_data = []
neighbor_sent = ''
for sentence in valid_data['sentence']:
    for keyword in keywords_list:
        ss = ''
        if keyword in sentence:
            temp_result = re.finditer(keyword,sentence)
            for temp in temp_result:
                start_id, end_id = temp.span()
                if start_id > 10 and end_id < len(sentence) - 9:
                    neighbor_sent = sentence[start_id - 10:end_id + 9]
                elif start_id < 10:
                    neighbor_sent = sentence[:20]
                elif end_id > len(sentence) - 9:
                    neighbor_sent = sentence[-20:]
                s = '{}-{}({}):{}'.format(start_id,end_id,sentence[start_id:end_id],get_entityid(neighbor_sent))
                ss += s + '|'
            result_data.append(ss[:-1])

result = pd.DataFrame(result_data)
print(pd.concat([result,valid_data['sentence']],axis=1))
