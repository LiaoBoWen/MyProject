from pyhanlp import *
"""
HanLP开启命名实体识别

"""

# 音译人名示例
CRFnewSegment = HanLP.newSegment("crf")
term_list = CRFnewSegment.seg("译智社的田丰要说的是这只是一个hanlp命名实体识别的例子")
print(term_list)


print("\n========== 命名实体开启与关闭对比试验 ==========\n")
sentences =[
    "北川景子参演了林诣彬导演的《速度与激情3》",
    "林志玲亮相网友:确定不是波多野结衣？",
    "龟山千广和近藤公园在龟山公园里喝酒赏花",
]
# 通过HanLP 进行全局设置,但是部分分词器本身可能不支持某项功能
# 部分分词器本身对某些命名实体识别效果较好
# HanLP.Config.japaneseNameRecognize = False

viterbiNewSegment = HanLP.newSegment("viterbi").enableJapaneseNameRecognize(True)
CRFnewSegment_new = HanLP.newSegment("crf").enableJapaneseNameRecognize(True)
# segSentence
# CRFnewSegment_2.seg2sentence(sentences)
for sentence in sentences:
    print("crf : ",CRFnewSegment.seg(sentence))
    print("crf_new : ",CRFnewSegment_new.seg(sentence))
    print("viterbi : ",viterbiNewSegment.seg(sentence))





# 中文人名识别
def demo_chinese_name_recognition(sentences):
    segment = HanLP.newSegment().enableNameRecognize(True);
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
        print([i.word for i in term_list])


sentences = [
    "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。",
    "武大靖创世界纪录夺冠，中国代表团平昌首金",
    "区长庄木弟新年致辞",
    "朱立伦：两岸都希望共创双赢 习朱历史会晤在即",
    "陕西首富吴一坚被带走 与令计划妻子有交集",
    "据美国之音电台网站4月28日报道，8岁的凯瑟琳·克罗尔（凤甫娟）和很多华裔美国小朋友一样，小小年纪就开始学小提琴了。她的妈妈是位虎妈么？",
    "凯瑟琳和露西（庐瑞媛），跟她们的哥哥们有一些不同。",
    "王国强、高峰、汪洋、张朝阳光着头、韩寒、小四",
    "张浩和胡健康复员回家了",
    "王总和小丽结婚了",
    "编剧邵钧林和稽道青说",
    "这里有关天培的有关事迹",
    "龚学平等领导说,邓颖超生前杜绝超生",]
demo_chinese_name_recognition(sentences)

print("\n========== 中文人名 基本默认已开启 ==========\n")
print(CRFnewSegment.seg(sentences[-3]))






# 音译人名识别
sentences = [
    "一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。",
    "世界上最长的姓名是简森·乔伊·亚历山大·比基·卡利斯勒·达夫·埃利奥特·福克斯·伊维鲁莫·马尔尼·梅尔斯·帕特森·汤普森·华莱士·普雷斯顿。",
]

segment = HanLP.newSegment().enableTranslatedNameRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

print("\n========== 音译人名 默认已开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))


# 日语人名识别
def demo_japanese_name_recognition(sentences):
    segment = HanLP.newSegment().enableJapaneseNameRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
        print([i.word for i in term_list])


sentences = [
    "北川景子参演了林诣彬导演的《速度与激情3》",
    "林志玲亮相网友:确定不是波多野结衣？",
    "龟山千广和近藤公园在龟山公园里喝酒赏花",
]
demo_japanese_name_recognition(sentences)
print("\n========== 日文人名 标准分词器默认未开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))





# 演示数词与数量词识别
sentences = [
    "十九元套餐包括什么",
    "九千九百九十九朵玫瑰",
    "壹佰块都不给我",
    "９０１２３４５６７８只蚂蚁",
    "牛奶三〇〇克*2",
    "ChinaJoy“扫黄”细则露胸超2厘米罚款",
]

StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")

StandardTokenizer.SEGMENT.enableNumberQuantifierRecognize(True)
for sentence in sentences:
    print(StandardTokenizer.segment(sentence))

print("\n========== 演示数词与数量词 默认未开启 ==========\n")
CRFnewSegment.enableNumberQuantifierRecognize(True)
print(CRFnewSegment.seg(sentences[0]))





# 机构名识别
sentences = [
    "我在上海林原科技有限公司兼职工作，",
    "我经常在台川喜宴餐厅吃饭，",
    "偶尔去开元地中海影城看电影。",
]

Segment = JClass("com.hankcs.hanlp.seg.Segment")
Term = JClass("com.hankcs.hanlp.seg.common.Term")

segment = HanLP.newSegment().enableOrganizationRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)

print("\n========== 机构名 标准分词器已经全部关闭 ==========\n")
print(CRFnewSegment.seg(sentences[0]))

segment = HanLP.newSegment('crf').enableOrganizationRecognize(True)


# 地名识别
def demo_place_recognition(sentences):
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
        print([i.word for i in term_list])      # 通过i.nature 获取词性


sentences = ["蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"]
demo_place_recognition(sentences)

print("\n========== 地名 默认已开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))





