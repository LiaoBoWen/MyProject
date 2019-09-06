import jieba
import jieba.analyse
import jieba.posseg as pseg     # 词性标注

# jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数 不支持windows

s = "此外，公司拟对全资子公司吉林欧亚置业有限公司增资4.3亿元，增资后，吉林欧亚置业注册资本由7000万元增加到5亿元。吉林欧亚置业主要经营范围为房地产开发及百货零售等业务。目前在建吉林欧亚城市商业综合体项目。2013年，实现营业收入0万元，实现净利润-139.13万元。"
jieba.suggest_freq('欧亚置业',True)
# todo 基于TF-IDF算法的关键词抽取
for x, w in jieba.analyse.extract_tags(s, withWeight=True):
    print('%s %s' % (x, w))

# todo 基于textRank算法的关键词抽取
print('=' * 20 )
jieba.add_word('欧亚置业')
for x, w in jieba.analyse.textrank(s,withWeight=True):
    print("{} {}".format(x,w))


words = pseg.cut('我爱北京天安门！')
for word, frag in words:
    print('{} {}'.format(word,frag))

