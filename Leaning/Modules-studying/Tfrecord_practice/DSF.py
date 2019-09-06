with open('test.txt','w') as f:
    # for i in range(100000000):
    #     print('\r{}'.format(i),end='')
    #     f.write('0 6.0,7.0,8.0,9.0,10.0 chengdu,chongqing\n')
    f.writelines(['{} 6.0,7.0,8.0,9.0,10.0 chengdu,chongqing\n'.format(i) for i in range(1000000)])