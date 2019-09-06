import os
import time
import json
import pickle
import logging
import requests
from lxml import etree


from intoDb import Mongo

logger = logging.getLogger('log_111')

logger.setLevel(logging.INFO)  # 这是下限

# 新建，格式，添加格式
stream_handler = logging.StreamHandler()  # sys.stdout
file_headler = logging.FileHandler('log.log', encoding='utf8')
# stream_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s[%(levelname)s]%(lineno)d||%(module)s||%(message)s', datefmt="%d %b %Y %X")
stream_handler.setFormatter(formatter)
file_headler.setFormatter(formatter)
logger.addHandler(file_headler)
logger.addHandler(stream_handler)


class Spider:
    def __init__(self, sleep_every_stid=20, sleep_every_page=20,speed=8):
        # sleep time
        self.sleep_every_stid = sleep_every_stid
        self.sleep_every_page = sleep_every_page
        self.speed = speed


        self.header = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36'}
        self.Mongo = Mongo()

    def spider_main(self, team_name, stid):
        '''
        进行数据的爬取
        :param stid:
        :return:
        '''
        logger.info('stid:{}'.format(stid))
        counter = 0

        if not os.path.exists('./saved_flag_/{}'.format(team_name)):
            os.makedirs('./saved_flag_/{}'.format(team_name))

        if not os.path.exists('./saved_flag_/{}/{}'.format(team_name,stid)):
            pickle.dump([],open('./saved_flag_/{}/{}'.format(team_name,stid),'wb'))
        saved_in_page = pickle.load(open('./saved_flag_/{}/{}'.format(team_name,stid),'rb'))

        for i_ in range(60, 0, -1):
            # 如果爬过该页，跳过
            if i_ in saved_in_page:
                continue
            # 达到今天的任务量，跳出
            if counter  >= self.speed:
                return 'break'
            # 开始爬取每页数据        print(stids)

            result = []
            logger.info('第{}页'.format(i_))
            Europ = {}
            try:
                resp = requests.get('http://liansai.500.com/index.php?'
                                'c=score&a=getmatch&stid={}&round={}'.format(stid, i_), headers=self.header)
                text = resp.json()
            except:
                time.sleep(1)
                continue
            for fid in text:
                # 进入初盘
                start_url = requests.get(
                    'http://odds.500.com/fenxi1/inc/ajax.php?_={}&t=oupei&cid=5&fid%5B%5D={}&p_t=1&sid%5B%5D=5&r=1'.format(
                        int(time.time() * 1000), fid['fid']), headers=self.header)
                start_url_1 = requests.get('http://odds.500.com/fenxi/yazhi-{}.shtml'.format(fid['fid']),
                                           headers=self.header)

                try:
                    start_data = start_url.json()
                    start_url_1.encoding = 'gb2312'
                    start_data_1_html = etree.HTML(start_url_1.text)
                    for i in range(1, 6):
                        temp = \
                            start_data_1_html.xpath('//div[@id="table_cont"]/table/tr[{}]/td[2]/p/a/@title'.format(i))[0]
                        if temp == '澳门':
                            start_data_1 = start_data_1_html.xpath(
                                '//div[@id="table_cont"]/table/tr[{}]/td[5]/table/tbody/tr/td/text()'.format(i))
                            break

                    # 获取两个水位
                    water_1 = float(start_data_1[0])
                    water_2 = float(start_data_1[2])


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

                    team_b = \
                    html.xpath('//div[@class="M_box"]/div[@class="M_content"]/div[@class="team_b"]/table/tbody')[0]

                    team_b_since_all_score = team_b.xpath('./tr[2]/td/text()')[-1]
                    team_b_since_host_score = team_b.xpath('./tr[3]/td/text()')[-1]
                    team_b_since_guest_score = team_b.xpath('./tr[4]/td/text()')[-1]
                    team_a_since_Statistics = team_a.xpath('./tr[2]/td/text()')[1:]
                    team_b_since_Statistics = team_b.xpath('./tr[2]/td/text()')[1:]

                    # print(team_b_since_guest_score)   测试正确性

                    if fid['hscore'] == None:
                        print('pass')
                        continue
                    Europ[fid['fid']] = {'stime': fid['stime'], 'hname': fid['hname'], 'gname': fid['gname'],
                                         'hscore': fid['hscore'], 'gscore': fid['gscore'],
                                         'hstanding': fid['hstanding'], 'gstanding': fid['gstanding'],
                                         'conpetition': team_name,
                                         'team_a_since_all_score':team_a_since_all_score,
                                         'team_b_since_all_score':team_b_since_all_score,
                                         'team_a_since_host_score':team_a_since_host_score,
                                         'team_b_since_host_score':team_b_since_host_score,
                                         'team_a_since_guest_score':team_a_since_guest_score,
                                         'team_b_since_guest_score':team_b_since_guest_score,
                                         'fid':fid['fid'],
                                         'team_a_since_Statistics': team_a_since_Statistics,
                                         'team_b_since_Statistics': team_b_since_Statistics}    # 修改增加了6个胜率

                    Europ[fid['fid']].update(start_data[fid['fid']])
                    Europ[fid['fid']].update(
                        {'handline': start_data_1[1], 'water1': water_1, 'water_2': water_2} if start_data[fid['fid']][
                            'WIN'] else {'handline': '', 'water1': '', 'water_2': ''})

                except Exception as e:
                    logger.error(e)
            logger.info('{} {}'.format(len(Europ), Europ))
            result.extend(Europ.values())


            # 标记过程
            if len(result) >= 6:
                logger.info('{}已保存{}条数据\n'.format(stid,len(result)))
                saved_in_page = pickle.load(open('./saved_flag_/{}/{}'.format(team_name,stid), 'rb'))
                saved_in_page.append(i_)
                pickle.dump(saved_in_page, open('./saved_flag_/{}/{}'.format(team_name,stid), 'wb'))
                # 插入数据库
                counter += 1
                self.Mongo.insert(result)
            time.sleep(self.sleep_every_page)

    def spider1(self, team_name, stids=None):
        '''
        获取每个date里面的数据
        :param stids: date_ids
        :return: None
        '''
        for stid in stids:
            # if stid in saved_stids:
            #     continue
            sign = self.spider_main(team_name=team_name, stid=stid)
            if sign == 'break':
                break
            time.sleep(self.sleep_every_stid)

    def diff_date(self, name, team_dict):
        '''
        :param name: 联赛的名称
        :return: 返回该联赛dateList
        '''
        get_stid_page = requests.get(team_dict[name], headers=self.header).text
        stids_html = etree.HTML(get_stid_page)
        raw_stids = stids_html.xpath('//ul[@class="ldrop_list"]/li/a/@href')

        raw_url = 'http://liansai.500.com'
        stids = []
        for i in raw_stids:
            resp = requests.get(raw_url + i, headers=self.header).text
            stids.append(etree.HTML(resp).xpath('//div[@id="season_match_round"]/div/a/@href')[0].split('-')[-1][:-1])
            time.sleep(3)
        logger.info(stids)

        if not os.path.exists('./data{}'.format(name)):
            os.mkdir('./data{}'.format(name))

        if os.path.exists(config['date_path'].format(name)):
            with open(config['date_path'].format(name), 'rb') as date_f:
                dateList_saved = pickle.load(date_f)

            if dateList_saved != stids:
                with open(config['date_path'].format(name), 'wb') as date_f:
                    pickle.dump(stids, date_f)
                logger.info('日期数据更新完毕.')
                stids = dateList_saved

        else:
            with open(config['date_path'].format(name), 'wb') as date_f:
                pickle.dump(stids, date_f)
            logger.info('日期数据更新完毕.')

        # 创建page标记的文件夹
        if not os.path.exists('./saved_flag_/{}'.format(name)):
            os.makedirs('./saved_flag_/{}'.format(name))

        self.spider1(name, stids)

    # 测试::获取每个联赛的stids用
    def test(self,name,team_dict):
        get_stid_page = requests.get(team_dict[name], headers=self.header).text
        stids_html = etree.HTML(get_stid_page)
        raw_stids = stids_html.xpath('//ul[@class="ldrop_list"]/li/a/@href')

        raw_url = 'http://liansai.500.com'
        stids = []
        for i in raw_stids:
            logger.info(raw_url + i)
            resp = requests.get(raw_url + i, headers=self.header).text
            stids.append(etree.HTML(resp).xpath('//div[@id="season_match_round"]/div/a/@href')[0].split('-')[-1][:-1])

        logger.info('{} {}'.format(name,stids))

def test_2(team_name,stids,pages_num):
    if not os.path.exists('./saved_flag_/{}'.format(team_name)):
        os.makedirs('./saved_flag_/{}'.format(team_name))


    for stid in stids:
        if not os.path.exists('./saved_flag_/{}/{}'.format(team_name,stid)):
            pickle.dump([],open('./saved_flag_/{}/{}'.format(team_name,stid),'wb'))

        saved_in_stids = pickle.load(open('./saved_flag_/{}/{}'.format(team_name,stid),'rb'))
        saved_in_stids.extend(range(1,pages_num + 1))
        pickle.dump(saved_in_stids,open('./saved_flag_/{}/{}'.format(team_name,stid),'wb'))


if __name__ == '__main__':

    # 创建页面标记文件夹
    if not os.path.exists('./saved_flag_'):
        os.makedirs('./saved_flag_')

    while True:
        with open('config.json','r') as f_json:
            config = json.load(f_json)


        # Main
        try:
            C_spider = Spider(sleep_every_stid=config['sleep_every_stid'], sleep_every_page=config['sleep_every_page'],speed=config['speed'])
            logger.info('Start .')
            with open('team_dict.json', 'r') as f:
                team_dict = json.load(f)
            names = team_dict.keys()
            for name in names:
                stids = C_spider.diff_date(name=name, team_dict=team_dict)



            time.sleep(60 * 60 * 24 * config['every_day'])
            logger.info('Sleeping ....')



        except requests.exceptions.ConnectionError:
            logger.error('requests.exceptions.ConnectionError')
            time.sleep(20)

# 五大联赛 = 五大联赛和英超连接之间的关系
# 获取年月的id = html内容捕获
# 最后的定时功能 = 智能模块初级阶段