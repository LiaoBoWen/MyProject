import fool
from entity_unity import init, main_extract
from collections import defaultdict

# inputs = ['昨天我在八一广场旁 边的电信营业厅','购买了中国移动的股票，希望可以和江西的小伙伴们一起去长江赚点钱']
# words, ners = fool.analysis(inputs)
# print(words) # 返回【【（单词，词性）（）（）】】
# print(ners) # 返回【【（开始下标，结束下标，单词）（）（）】】


data = ['今天李大钊在江西银行买了一打代券，江西银行股份有限公司很开心']
citys, d_delete, stop_words = init()
ner_dict = defaultdict(list)
init_id = 1001
ner_id = {}


'''提取目的实体，并归类到统一后的实体字典中'''
for i in range(len(data)):
    words, ners = fool.analysis(data[i])
    for start, end, n_type, word in ners[0]:
        if n_type == 'company' or n_type == 'person':
            ner_result = main_extract(word, stop_words, d_delete, citys)
            ner_dict[ner_result].append(word)
            if word not in ner_id:
                ner_id[ner_result] = init_id
                init_id += 1
    data[i] = '{}ner_{}_{}'.format(data[i][:start], ner_id[ner_result], data[i][end - 1:])


print(ner_id)
