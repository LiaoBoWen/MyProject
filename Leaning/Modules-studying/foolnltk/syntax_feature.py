from pyltp import Parser, Segmentor, Postagger
from itertools import combinations
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import re

'''句法分析： 参考特征：1、企业实体间距离 2、企业实体间句法距离 3、企业实体分别与关键触发词的距离 4、字体的依存关系类别'''

postagger = Postagger()
postagger.load_with_lexicon('./ltp_data_v3.4.0/pos.model')
segmentor = Segmentor()
segmentor.load_with_lexicon('./ltp_data_v3.4.0/cws.model')


def shortest_path(arcs_ret, source, target):
    '''
    求出两个词最短依存句法路径， 不存在的路径返回-1
    :param arcs_ret: 句法分析结果表格
    :param source: 实体1
    :param target: 实体2
    '''
    Graph = nx.DiGraph()
    # 为这个网络添加节点
    for i in list(arcs_ret.index):
        Graph.add_node(i)
        # TODO 在网络中添加带权重的边（无向边）
        Graph.add_edge(arcs_ret.loc[i, 2], i)
    # TODO 转化成无向图
    Graph.to_undirected()

    try:
        shortest_distance = nx.shortest_path_length(Graph, source=source, target=target)
        return shortest_distance
    except:
        return -1


def parse(sentence):
    '''对语句进行句法分析， 并返回句法结果'''
    temp_ner_dict = {}
    num_list = ['一','二','三','四','五','六','七','八','九','十']

    for i, ner in enumerate(list(set(re.findall(r'ner_\d{4}_',sentence)))):
        try:
            temp_ner_dict[num_list[i] + '号企业'] = ner
        except IndexError:
            print('替换出错！')
        sentence = sentence.replace('ner','{}号企业'.format(num_list[i]))

    words = segmentor.segment(sentence)
    tags = postagger.postag(words)
    parser = Parser()
    parser.load('./ltp_data_v3.4.0/parser.model')
    arcs = parser.parse(words, tags)
    arcs_lst = list(map(list, zip(*[[arc.head, arc.relation] for arc in arcs])))

    # 句法分析结果
    parse_result = pd.DataFrame([[a, b, c, d] for a, b, c, d in zip(list(words),  list(tags), arcs_lst[0])],
                                index=range(1, len(words) + 1))

    # 释放模型
    parser.release()

    # 投资关系关键词
    key_words = ['收购','竞拍','转让','扩张','注资','整合','并入','竞购','竞买','支付','收购价','收购价格','承购','购得','购进','购入','买进','买入','赎买','购销','议购','函购','函售','抛售','售卖','销售','转售']

    # TODO 提取关键词和对应句法关系提取特征

    ner_index_list = []
    keyword_index_list = []

    for idx in parse_result.index:
        if parse_result.loc[idx, 0].endswith('号企业'):
            ner_index_list.append(idx)

        if parse_result.loc[idx, 0] in key_words:
            keyword_index_list.append(idx)

    # 1、句子中关键词的数量
    parse_feature1 = len(keyword_index_list)
    # 2、若关键词存在
    # 初始化判断与关键词有直接关系的“X号企业”句法类型为“S"的数量
    parse_feature2 = 0
    # 初始化判断与关键词有直接关系的”X号企业“句法类型为”OB“的数量
    parse_feature3 = 0

    # 遍历出现在句子中的关键词索引
    for i in keyword_index_list:
        for j in ner_index_list:
            if parse_result.loc[j, 3].startswith('S') or parse_result.loc[j, 3].endswith('OB'):
                if parse_result.loc[i, 2] == j:
                    parse_feature2 += 1
                if parse_result.loc[j, 2] == i:
                    parse_feature3 += 1
    # 3 实体与关键词之间距离的平均值， 最大值和最小值
    ner_keyword_pair_list = [(ner_index, keyword_index) for ner_index in ner_index_list for keyword_index in keyword_index_list]
    ner_keyword_distance_list = [abs(pair[0] - pair[1] for pair in ner_keyword_pair_list)]

    parse_feature4 = np.mean(ner_keyword_distance_list) if ner_keyword_distance_list else 0
    parse_feature5 = max(ner_keyword_distance_list) if ner_keyword_distance_list else 0
    parse_feature6 = min(ner_keyword_distance_list) if ner_keyword_distance_list else 0

    # 4、实体与关键词之间句法距离的平均值，最大值和最小值
    ner_keyword_parse_distance_list = [shortest_path(parse_result, pair[0], pair[1]) for pair in ner_keyword_pair_list]

    parse_feature7 = np.mean(ner_keyword_parse_distance_list) if ner_keyword_parse_distance_list else 0
    parse_feature8 = max(ner_keyword_parse_distance_list) if ner_keyword_parse_distance_list else 0
    parse_feature9 = min(ner_keyword_parse_distance_list) if ner_keyword_parse_distance_list else 0

    # 5、实体与关键词之间句法距离的平均值， 最大值与最小值
    ner_pair_list = list(combinations(ner_index_list, 2))
    ner_distance_list = [abs(pair[0] - pair[1]) for pair in ner_pair_list]

    parse_feature10 = np.mean(ner_distance_list) if ner_distance_list else 0
    parse_feature11 = max(ner_distance_list) if ner_distance_list else 0
    parse_feature12 = min(ner_distance_list) if ner_distance_list else 0

    # 6、实体之间句法距离的平均值， 最大值和最小值
    ner_parse_distance_list = [shortest_path(parse_result, pair[0], pair[1]) for pair in ner_pair_list]

    parse_feature13 = np.mean(ner_parse_distance_list) if ner_parse_distance_list else 0
    parse_feature14 = max(ner_parse_distance_list) if ner_parse_distance_list else 0
    parse_feature15 = min(ner_parse_distance_list) if ner_parse_distance_list else 0

    return [parse_feature1, parse_feature2, parse_feature3, parse_feature4, parse_feature5, parse_feature6,
            parse_feature7, parse_feature8, parse_feature9, parse_feature10, parse_feature11, parse_feature12,
            parse_feature13, parse_feature14, parse_feature15]


# 汇总句法特征和TFIDF特征
def get_extract_feature(corpus):
    '''获取语料数据的额外特征'''
    # 存放所有语料数据额外特征
    extrac_feature_lst = []
    try:
        with tqdm(corpus) as t:
            for i in t:
                current_extra_features = parse(i)
                extrac_feature_lst.append(current_extra_features)
        return np.array(extrac_feature_lst)
    except KeyboardInterrupt:
        t.close()
        raise


