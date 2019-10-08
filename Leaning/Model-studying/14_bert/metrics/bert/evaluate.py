import json
import re
import nltk
import jieba
# nltk.download('punkt')

def mix_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ''
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != '':
                # ss = nltk.word_tokenize(temp_str)
                ss = jieba.cut(temp_str)
                segs_out.extend(ss)
                temp_str = ''
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != '':
        # ss = nltk.word_tokenize(temp_str)
        ss = jieba.cut(temp_str)
        segs_out.extend(ss)


    return segs_out


def remove_punction(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
			   '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
			   '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1] [j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p -mmax:p], mmax


def evaluate(truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in truth_file['data']:
        for para in instance['paragraphs']:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                query_text = qas['question'].strip()
                answers = [x['text'] for x in qas['answers']]

                if query_id not in prediction_file:
                    print('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_file[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100. * f1 / total_count
    em_score = 100. * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, predictions):
    f1_score = []
    for ans in answers:
        ans_segs = mix_segmentation(ans, rm_punc=True)
        prediction_segs = mix_segmentation(predictions, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_score.append(0)
            continue
        precision = 1. * lcs_len / len(prediction_segs)
        recall = 1. * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_score.append(f1)
    return max(f1_score)


def calc_em_score(answers, predictions):
    em = 0
    for ans in answers:
        ans_ = remove_punction(ans)
        predictions_ = remove_punction(predictions)
        if ans_ == predictions_:
            em = 1
            break
    return em


if __name__ == '__main__':
    '''貌似计算evaluate有点问题'''
    prediction = json.load(open('CMRC_output/predictions.json'))
    answer = json.load(open('/media/liao/Data/temp_data/squad_style_data/cmrc2018_dev.json'))
    f1, em, total, skip = evaluate(answer, prediction)
    print(f1, em, total, skip)