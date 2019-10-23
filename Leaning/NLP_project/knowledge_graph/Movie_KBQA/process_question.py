import jieba.posseg as psg
import jieba
import re
from classifier_question import Classifier
from template_question import Question_template

class Question:
    '''项目主干'''
    def __init__(self):
        self.classifier = Classifier()

        with open('./question/question_classification.txt', 'r', encoding='utf-8') as f:
            question_mode_list = f.read().splitlines()
        self.question_mode_dict = {}

        for mode in question_mode_list:
            id, sent = mode.strip().split(":")
            self.question_mode_dict[int(id)] = sent.strip()

        self.question_template = Question_template()


    def question_process(self, question):
        # 词性标注
        self.pos_question = self.question_posseg(question)
        # 模板归类
        self.question_template_id_str = self.get_question_template()
        # 利用cypher语句使用模板类得到答案
        self.answer = self.query_template()

        return self.answer


    def question_posseg(self, question):
        '''返回["word/flag"]的格式'''
        jieba.load_userdict('./question/userdict3.txt')
        processed_question = re.sub(r'[\s+.!/_,$%^&*()+"\']', '', question)
        question_seged = psg.cut(processed_question)
        self.processed_question = processed_question

        result = []
        self.words, self.flags = [], []
        for w in question_seged:
            result.append('{}/{}'.format(w.word, w.flag))
            word, flag = w.word, w.flag
            self.words.append(word.strip())
            self.flags.append(flag)

        assert len(self.flags) == len(self.words)
        print('word posseg: ',result)
        return result


    def get_question_template(self):
        for item in ['nr','nm','ng']:
            while item in self.flags:
                idx = self.flags.index(item)
                # 替换句子中特征性word
                self.words[idx] = item
                self.flags[idx] = item + 'ed'

        replaced_question = ''.join(self.words)
        print('抽象后的句子:', replaced_question)
        predict_class = self.classifier.predict(replaced_question)
        print('所属类别id:', predict_class)
        question_template = self.question_mode_dict[predict_class]
        print('问题模板:',question_template)
        return  predict_class


    def query_template(self):
        '''根据问题模本的具体内容,构造cypher语句,查询'''
        # try:
        answer = self.question_template.answer_question(self.pos_question, self.question_template_id_str)
        # except:
        #     answer = ('Don\'t know ...')
        return answer


if __name__ == '__main__':
    question = Question()
    question.question_process('李连杰的出身日期')
    answer = question.query_template()
    print(answer)