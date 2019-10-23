# https://github.com/zhangyunxing37/knowledge_grapy-kbqa_demo
# 至于识别,可以使用tfidf进行,之后使用分类模型(NB)进行判别即可

import os
import re
import ahocorasick
import synonyms
import jieba
import numpy as np

class Entity_extractor:
    def __init__(self):
        self.feature_path = 'data/feature'
        self.operate_path = 'data/operate'
        self.operate1_path = 'data/operate1'
        self.stop_word_path = 'data/stopword.txt'
        self.vocab_path = 'data/vocab'
        with open(self.feature_path, encoding='utf8') as f:
            self.feature_lst = f.read().splitlines()
        with open(self.operate_path, encoding='utf8') as f:
            self.operate_lst = f.read().splitlines()
        with open(self.operate1_path, encoding='utf8') as f:
            self.operate1_lst = f.read().splitlines()
        with open(self.stop_word_path, encoding='utf-8') as f:
            self.stop_word = f.read().splitlines()


        self.feature_tree = self.build_actree(self.feature_lst)
        self.operate_tree = self.build_actree(self.operate_lst)
        self.operate1_tree = self.build_actree(self.operate1_lst)


    def build_actree(self, wordlist):
        actree = ahocorasick.Automaton()
        for i, word in enumerate(wordlist):
            actree.add_word(word, (i, word))
        actree.make_automaton()
        return actree



    def full_match(self,question):
        '''使用ahocorasick进行全局搜索是否含有所定义的word'''
        self.result = {}
        for i in self.feature_tree.iter(question):
            if 'feature' not in self.result:
                self.result['feature'] = [i[1][1]]
            else:
                self.result['feature'].append(i[1][1])

        for i in self.operate_tree.iter(question):
            if 'operate' not in self.result:
                self.result['operate'] = i[1][1]
            else:
                self.result['operate'].append(i[1][1])

        for i in self.operate1_tree.iter(question):
            if 'operate1' not in self.result:
                self.result['operate1'] = i[1][1]
            else:
                self.result['operate1'].append(i[1][1])

        if 'result' not in self.result:
            self.find_sim_word(question, 'result')

        if 'operate' not in self.result:
            self.find_sim_word(question, 'operate')

        if 'operate1' not in self.result:
            self.find_sim_word(question, 'operate1')

    def find_sim_word(self, question, flag):
        question = re.sub(r'[.,。，？！?!]', ' ', question)
        question = question.strip()
        jieba.load_userdict(self.vocab_path)

        words = [word for word in jieba.cut(question) if word not in self.stop_word]
        temp = {'feature':self.feature_lst, 'operate':self.operate_lst, 'operate1':self.operate1_lst}
        entities = temp[flag]
        for word in words:
            scores = self.similarity(word, entities)
            if scores:
                self.result[flag] = scores[0][0]


    def similarity(self, word, entities, flag):
        '''利用synonyms'compare计算score与编辑距离做平均得到最后的衡量指数'''
        scores = []
        word_len = len(word)
        for entity in entities:
            temp = []
            entity_len = len(entity)
            try:
                score = synonyms.compare(word, entity)   # 该相似度利用余弦相似度与编辑距离同时计算
                if score:
                    temp.append(score)
            except:
                pass

            score2 = 1 - self.edit_distance(word, entity) / (word_len + entity_len)
            if score2 > 0.5:
                temp.append(score2)

            score3 = sum(temp) / len(temp)      # 利用synonyms的compare得到的相似度和编辑距离得到的相似度做平均得到最后的相似度分数
            if score3 > 0.7:
                scores.append((entity, score3, flag))

        scores.sort(key=lambda x:x[1], reverse=True)
        return scores



    def edit_distance(self, source, target):
        sourcelen = len(source)
        targetlen = len(target)
        solution = [[0 for _ in range(targetlen + 1)] for _ in range(sourcelen + 1)]

        for i in range(targetlen + 1):
            solution[0][i] = i
        for i in range(sourcelen + 1):
            solution[i][0] = i

        for i in range(1, sourcelen + 1):
            for j in range(1, targetlen + 1):
                if source[i - 1] == target[j - 1]:
                    solution[i][j] = solution[i - 1][j - 1]
                else:
                    solution[i][j] = min(solution[i][j - 1], solution[i - 1][j], solution[i - 1][j - 1]) + 1

        return solution[sourcelen][targetlen]

if __name__ == '__main__':
    test = Entity_extractor()
    print(test.edit_distance('liao', 'laao'))
