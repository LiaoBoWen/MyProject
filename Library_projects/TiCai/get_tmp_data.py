import json
import logging
import requests
import pymongo
import pickle
import time
import os
import pandas as pd
from lxml import etree


logger = logging.getLogger('log_111')

logger.setLevel(logging.INFO)  # 这是下限

# 新建，格式，添加格式
stream_handler = logging.StreamHandler()  # sys.stdout
formatter = logging.Formatter('%(asctime)s[%(levelname)s]%(lineno)d||%(module)s||%(message)s', datefmt="%d %b %Y %X")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)




class Mongo:
    # def __init__(self,ip='240434v2c3.wicp.vip',port=57967):
    def __init__(self,ip='192.168.2.149',port=27017):
        # 创建连接
        client = pymongo.MongoClient(ip,port)

        # 指定数据库
        self.db = client['lottery']

        # 指定集合名
        self.result = self.db['result']


class Spider:
    def __init__(self, sleep_every_stid=20, sleep_every_page=20,speed=8):
        # sleep time
        self.sleep_every_stid = sleep_every_stid
        self.sleep_every_page = sleep_every_page
        self.speed = speed

        self.header = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
        self.Mongo = Mongo()

    def spider_main(self, team_name, stid):
        '''
        进行数据的爬取
        :param stid:
        :return:
        '''
        logger.info('stid:{}'.format(stid))

        for i_ in range(38, 0, -1):
            logger.info('第{}页'.format(i_))

            try:
                resp = requests.get('http://liansai.500.com/index.php?'
                                'c=score&a=getmatch&stid={}&round={}'.format(stid, i_), headers=self.header)
                text = resp.json()
                print('##############  {}'.format(text))
            except:
                time.sleep(1)
                continue
            for fid in text:
                try:
                    # 爬取两队以往的胜率
                    response = requests.get('http://odds.500.com/fenxi/shuju-{}.shtml'.format(fid['fid']),headers=self.header)
                    response.encoding = 'gb2312'

                    html = etree.HTML(response.text)
                    # M_content 包含两个球队的赛前积分排名
                    team_a = \
                    html.xpath('//div[@class="M_box"]/div[@class="M_content"]/div[@class="team_a"]/table/tbody')[0]

                    # team_a_sincea_all_score 的结果为：比赛 胜 平 负 进 失 净 积分 排名 胜率
                    team_a_since_all_score = team_a.xpath('./tr[2]/td/text()')[-1]
                    team_a_since_host_score = team_a.xpath('./tr[3]/td/text()')[-1]
                    team_a_since_guest_score = team_a.xpath('./tr[4]/td/text()')[-1]
                    team_a_since_Statistics = team_a.xpath('./tr[2]/td/text()')[1:]

                    team_b = \
                    html.xpath('//div[@class="M_box"]/div[@class="M_content"]/div[@class="team_b"]/table/tbody')[0]

                    team_b_since_all_score = team_b.xpath('./tr[2]/td/text()')[-1]
                    team_b_since_host_score = team_b.xpath('./tr[3]/td/text()')[-1]
                    team_b_since_guest_score = team_b.xpath('./tr[4]/td/text()')[-1]
                    team_b_since_Statistics = team_b.xpath('./tr[2]/td/text()')[1:]

                    # print(team_b_since_guest_score)   测试正确性

                    if fid['hscore'] == None:
                        logger.info('pass')
                        continue
                    condition= {'stime': fid['stime'], 'hname': fid['hname'], 'gname': fid['gname'],
                                         'conpetition': team_name}
                                         # 以上为更新条件，一下为更新信息
                    new_data = {'team_a_since_all_score':team_a_since_all_score,
                                         'team_b_since_all_score':team_b_since_all_score,
                                         'team_a_since_host_score':team_a_since_host_score,
                                         'team_b_since_host_score':team_b_since_host_score,
                                         'team_a_since_guest_score':team_a_since_guest_score,
                                         'team_b_since_guest_score':team_b_since_guest_score,
                                         'team_a_since_Statistics':team_a_since_Statistics,
                                         'team_b_since_Statistics':team_b_since_Statistics,
                                         'fid':fid['fid']}    # 修改增加了6个胜率

                    logger.info(new_data)
                    self.Mongo.result.update_one(condition,{'$set':new_data})
                    state = self.Mongo.result.find(condition)
                    _ = list(state)
                    if _ != []:
                        for __ in _:
                            print('查找结果： ', __)
                            __.pop('_id')
                            test_list.append(__)
                            with open('temp_data.pkl','wb') as f:
                                pickle.dump(test_list,f)

                except Exception as e:
                    logger.error(e)
            time.sleep(self.sleep_every_page)

    def spider1(self, team_name, stids=None):

        for stid in stids:
            sign = self.spider_main(team_name=team_name, stid=stid)
            if sign == 'break':
                break
            time.sleep(self.sleep_every_stid)

    def diff_date(self, name):
    #     get_stid_page = requests.get("http://liansai.500.com/zuqiu-3822", headers=self.header).text
    #     stids_html = etree.HTML(get_stid_page)
    #     raw_stids = stids_html.xpath('//ul[@class="ldrop_list"]/li/a/@href')
    #     logger.info(raw_stids)
    #
    #     raw_url = 'http://liansai.500.com'
    #     stids = []
    #     for i in raw_stids:
    #         logger.info(i)
    #         try:
    #             resp = requests.get(raw_url + i, headers=self.header).text
    #             stids.append(
    #                 etree.HTML(resp).xpath('//div[@id="season_match_round"]/div/a/@href')[0].split('-')[-1][:-1])
    #             time.sleep(3)
    #         except:
    #             logger.error('Error in stids range!')
    #             time.sleep(5)
    #             resp = requests.get(raw_url + i, headers=self.header).text
    #             stids.append(
    #                 etree.HTML(resp).xpath('//div[@id="season_match_round"]/div/a/@href')[0].split('-')[-1][:-1])
    #             time.sleep(3)
    #     logger.info(stids)
        stids = ['649', '823', '967', '2573', '3270', '4030', '4794', '5428', '6118', '6832', '7471', '8658']
        self.spider1(name, stids)

if __name__ == '__main__':

    test_list = []
    while True:
        with open('config.json', 'r') as f_json:
            config = json.load(f_json)

        try:

            C_spider = Spider(sleep_every_stid=config['sleep_every_stid'], sleep_every_page=config['sleep_every_page'],speed=config['speed'])

            logger.info('Start .')

            stids = C_spider.diff_date(name='EC')

            time.sleep(60 * 60 * 24 * config['every_day'])
            logger.info('Sleeping ....')


        except requests.exceptions.ConnectionError:
            logger.error('requests.exceptions.ConnectionError')
            time.sleep(20)

    #############################################
    # with open('temp_data.pkl', 'rb') as f:
    #     all = pickle.load(f)
    #     text = pd.DataFrame(all)
    #     text.to_csv('data.csv',index=None)
    #############################################

'''
{'_id': ObjectId('5ca36d31db473f2d68008257'), 'DRAW': '3.25', 'LOST': '3.9', 'WIN': '1.85', 'gname': '西布罗姆维奇', 'gscore': '2.0', 'gstanding': '19', 'handline': '半球', 'hname': '埃弗顿', 'hscore': '2.0', 'hstanding': '11', 'stime': '2006-05-07 22:00', 'water1': '0.9', 'water_2': '0.95', 'conpetition': 'EC', 'winner': '平', 'fid': '117353', 'team_a_since_all_score': '38%', 'team_a_since_guest_score': '32%', 'team_a_since_host_score': '45%', 'team_b_since_all_score': '19%', 'team_b_since_guest_score': '6%', 'team_b_since_host_score': '32%'}

德甲 德乙 英超 法甲 法乙 挪超 日的超 日职 日职乙 西甲 英冠 意甲 瑞典超
'''