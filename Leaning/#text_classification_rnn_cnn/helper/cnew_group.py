import os

def _read_file(filename):
    '''读取一个文件并转换为一行'''
    with open(filename,'r',encoding='utf8') as f:
        return f.read().replace('\n','').replace('\t','').replace('\u3000','')

def save_file(dirname):
    '''
    将多个文件整合并存放到3个文件里面
    :param dirname: 原数据目录
    文件格式内容：类别\t内容
    :return:
    '''
    f_train = open('../data/cnews/cnew.train.txt','w',encoding='utf8')
    f_test = open('../data/cnews/cnew.test.txt','w',encoding='utf8')
    f_val = open('../data/cnews/cnew.val.txt','w',encoding='utf8')

    for category in os.listdir(dirname):
        cat_dir = os.path.join(dirname,category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir,cur_file)
            content = _read_file(filename)
            if count < 5000:
                f_train.write(category + '\t' + content + '\n' )
            elif count < 6000:
                f_test.write(category + '\t' + content + '\n')
            elif count < 7000:
                f_val.write(category + '\t' + content + '\n')
            count += 1
            else:
                break
        print('Finished: ',category)


    f_train.close()
    f_test.close()
    f_val.close()

if __name__ == '__main__':
    save_file('../data')
    print(len(open('../data/cnew/cnews.train.txt','r',encoding='utf8').readlines()))
    print(len(open('../data/cnew/cnews.test.txt','r',encoding='utf8').readlines()))
    print(len(open('../data/cnew/cnews.val.txt','r',encoding='utf8').readlines()))