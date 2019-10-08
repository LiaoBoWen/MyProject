import pandas as pd
from bert.run_squad import *

# with open('/media/liao/Data/temp_data/莱斯杯训练数据/test_data_r0.csv') as f:
    # data = pd.read_csv(f,chunksize=1)
    # # print(data.loc[78])
    # # for i in data.itertuples(name='QA'):
    # #     print(i)
    # #     break
    # for val in data:
    #     # print(val.columns)
    #     '''
    #     train:
    #     Index(['answer', 'bridging_entity', 'content1', 'content2', 'content3',
    #     'content4', 'content5', 'keyword', 'question', 'supporting_paragraph',
    #     'title1', 'title2', 'title3', 'title4', 'title5', 'question_id'],
    #     dtype='object')
    #    test:
    #    Index(['content1', 'content2', 'content3', 'content4', 'content5', 'keyword',
    #    'question', 'title1', 'title2', 'title3', 'title4', 'title5',
    #    'question_id'],
    #    dtype='object')
    #     '''
    #     break

import json
with open('/media/liao/Data/temp_data/squad_style_data/cmrc2018_train.json') as f:
    data = json.load(f)
    # print(data['data'][0]['paragraphs'][0]['qas'][0])
    print(len(data['data']))

for i in range(len(data['data'])):
    if 'DEV_1112' in data['data'][i]['paragraphs'][0]['qas'][0]['id']:
        pass
        # print(data['data'][i])
# print(data['data'][1112])
data_example = {'paragraphs':
         [{'id': 'DEV_0',
           'context': '《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品',
           'qas': [{'question': '《战国无双3》是由哪两个公司合作开发的？', 'id': 'DEV_0_QUERY_0', 'answers': [{'text': '光荣和ω-force', 'answer_start': 11}, {'text': '光荣和ω-force', 'answer_start': 11}, {'text': '光荣和ω-force', 'answer_start': 11}]}
                  ,{'question': '男女主角亦有专属声优这一模式是由谁改编的？', 'id': 'DEV_0_QUERY_1', 'answers': [{'text': '村雨城', 'answer_start': 226}, {'text': '村雨城', 'answer_start': 226}, {'text': '任天堂游戏谜之村雨城', 'answer_start': 219}]},
                   {'question': '战国史模式主打哪两个模式？', 'id': 'DEV_0_QUERY_2', 'answers': [{'text': '「战史演武」&「争霸演武」', 'answer_start': 395}, {'text': '「战史演武」&「争霸演武」', 'answer_start': 395}, {'text': '「战史演武」&「争霸演武」', 'answer_start': 395}]}]}],
     'id': 'DEV_0', 'title': '战国无双3'}


# with open('CMRC_output/predictions.json') as f:
#     predict = json.load(f)
    # print(predict)


with open('laisi_train_data.csv') as f:
    data = pd.read_csv(f)
    print(data.loc[334].content4)
    print(data.loc[334])
    # print(data.loc[3784]['content1'])
    # print(data.loc[3784]['content2'])
    # print(data.loc[3784]['content3'])
    # print(data.loc[3784]['content4'])
    # print(data.loc[3784]['content5'])
    # print(data.loc[3784]['answer'])

    # data.drop(data['answer'][data.loc[:,'answer'].apply(lambda x: len(x.strip().split('@')) - 5 > 0)].index, inplace=True)

    # print(data['answer'][data.loc[:,'answer'].apply(lambda x: len(x.strip().split('@')) - 5) > 0])

with open('/media/liao/Data/temp_data/莱斯杯训练数据/test_data_r0.csv') as  f:
    data = pd.read_csv(f)
    print(data.loc[0])
    print(data.shape)