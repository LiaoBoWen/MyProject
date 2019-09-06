class Params:
    vocab_size = 32284
    batch_size = 32
    lr = 0.00005
    num_units = 512
    num_heads = 8
    num_blocks = 6
    dropout_rate = 0.5
    smooth = 0.1
    warmup_step = 4000
    epochs = 50
    maxlen = 20
    dim_feed_forword =  2048
    vocab_path = '../data/xiaohuangji50w_nofenci.conv'