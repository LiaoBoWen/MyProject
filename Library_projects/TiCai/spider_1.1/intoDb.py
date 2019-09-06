import pymongo
import pandas as pd


class Mongo:
    # def __init__(self,ip='240434v2c3.wicp.vip',port=57967):
    def __init__(self,ip='192.168.2.149',port=27017):
        # 创建连接
        client = pymongo.MongoClient(ip,port)

        # 指定数据库
        self.db = client['lottery']

        # 指定集合名
        self.result = self.db['neoFootballData']


    def insert(self,datas):
        # 插入数据
        for data in datas:
            try:
                state = self.result.insert_one(data)
            except:
                pass
       # print(state.acknowledged)
       # print('Insert Finished.')

    def search(self,condition):
        test = self.result.find(condition)
        for i in test:
            print(i)

    def delete(self,condition):
        self.result.delete_many(condition)
        print('finish delete .')

if __name__ == '__main__':
    mongo = Mongo()


    mongo.delete({'conpetition':'DE'})
    # data = mongo.result.find()
        # for i in data:
        #     print(data)
    test = mongo.search({'conpetition':'DE'})
    print(test)
    # data = pd.read_csv('result_.csv',header=0,names=['1','2','3','4','5','6','7','8','9','10','11'])
    # all_data = data.to_dict(orient='records')