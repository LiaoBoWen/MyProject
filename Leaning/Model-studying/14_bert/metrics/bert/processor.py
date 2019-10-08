import re
import pandas as pd

def remove_multi_answer(raw_data_path, new_data_path):
    with open(raw_data_path) as f:
        data = pd.read_csv(f)
        # 去除多个答案的Example
        data.drop(data['answer'][data['answer'].apply(
            lambda x: len(re.split(r'@content\d@',x)) - 3
        )].index, inplace=True)

        data.content1 = data.content1.apply(lambda x: x.replace('  ','\t'))
        data.content2 = data.content2.apply(lambda x: x.replace('  ','\t'))
        data.content3 = data.content3.apply(lambda x: x.replace('  ','\t'))
        data.content4 = data.content4.apply(lambda x: x.replace('  ','\t'))
        data.content5 = data.content5.apply(lambda x: x.replace('  ','\t'))
        data.to_csv(new_data_path,index=None)


if __name__ == '__main__':
    remove_multi_answer('/media/liao/Data/temp_data/莱斯杯训练数据/test_data_r0.csv','laisi_test_data.csv')
    with open('laisi_test_data.csv','r') as f:
        data = pd.read_csv(f)
        print(data.shape)