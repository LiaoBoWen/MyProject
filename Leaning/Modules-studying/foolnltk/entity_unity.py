import jieba
import re

def clean_word(seg, stop_word, d_delete):
    seg = [word for word in seg if word not in stop_word]
    seg = [word for word in seg if word not in d_delete]
    return seg

def citys_to_ahead(seg, citys):
    # 地点提前
    location = [place for place in seg if place in citys]
    seg = [word for word in seg if word not in citys]
    return location + seg

def main_extract(inputs, stop_words, d_delete, citys):
    seg = jieba.cut(inputs)
    seg = clean_word(seg, stop_words, d_delete)
    seg = citys_to_ahead(seg, citys)
    result = ''.join(seg)
    return result


def init(city_path='./data/dict/co_City_Dim.txt',
         company_suffix_path='./data/dict/company_suffix.txt',
         provinces_path='./data/dict/co_Province_Dim.txt',
         stop_path='./data/dict/stopwords.txt'):
    citys = open('./data/dict/co_City_Dim.txt',encoding='utf8')
    citys = citys.read().splitlines()
    # company_scope = open('./data/dict/company_business_scope.txt',encoding='utf8')
    # company_scope = company_scope.read().splitlines()
    company_suffix = open('./data/dict/company_suffix.txt',encoding='utf8')
    company_suffix = company_suffix.read().splitlines()
    provinces = open('./data/dict/co_Province_Dim.txt',encoding='utf8')
    provinces = provinces.read().splitlines()
    stop_words = open('./data/dict/stopwords.txt',encoding='utf8')
    stop_words = stop_words.read().splitlines()

    citys.extend(provinces)

    citys = set(citys)
    company_suffix = set(company_suffix)
    stop_words = set(stop_words)

    return citys, company_suffix, stop_words

if __name__ == '__main__':
    citys, d_delete, stop_words = init()
    inputs = '江西宜春银行股份有限公司'
    result = main_extract(inputs, stop_words, d_delete, citys)
    print(result)