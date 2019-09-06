import jieba.posseg as pseg

words =  pseg.cut('我廖博文,今天就算饿死，也不吃你的饭')  # 生成器，遍历就空

for word, flag in words:
    print(word,flag)
'''
的 uj
饭 n

'''

for word in words:
    print(word)      # 生成的是一种jieba的数据类型，单独输出的结果和分开输出的结果不一样
'''
的/uj
饭/n
'''