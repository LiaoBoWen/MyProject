config = {
    'sequence_length':300,
    'num_classes':10,
    'vocab_size':5000, # 字典大小
    'embedding_size':300,
    'l2_reg_lambda':.0,  # l2正则化参数
    'device':'/gpu:0',
    'batch_size':64,
    'num_epochs':10, # epoch 数目
    'evaluate_every':100,   # 每多少次进行一次验证数据的验证
    'checkpoint_every':100, # 每多少次保存一次模型
    'num_checkpoints':5,    # 最多保存模型的个数
    'allow_soft_placement': True, # 是否允许程序自动选择备用的device
    'log_device_placement':False, # 是否允许在终端打印日志文件
    'train_test_dev_rate':[0.97,0.02,0.01],
    'data_path':'./data/cnews.test.txt',
    'learning_rate':0.0007,
    'dropout_keep_prob':0.5,
    'vocab_path':'./vocabs',
    'rnn_hidden_size':512,     # RNN的隐层大小
    'text_hidden_size':512,     # FC层神经元个数
}