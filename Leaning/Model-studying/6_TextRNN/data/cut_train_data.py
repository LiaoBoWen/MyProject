def cut_data(file_path='train-zhihu4-only-title-all.txt'):
    with open(file_path,'r',encoding='utf8') as f:
        with open('miny-data.txt','w',encoding='utf8') as w:
            w.writelines([i for i in f.readlines()][:2000])

if __name__ == '__main__':
    cut_data()