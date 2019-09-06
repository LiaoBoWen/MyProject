class hyperparams:

    vocab_size = 20000
    batch_size = 64
    eval_batch_size = 128
    lr = 0.00002
    warmup_steps = 4000
    logdir = "./model"
    num_epochs = 100
    evaldir = "./model"
    num_units = 512     # 除以多头注意力的数量等于64，sqrt(64) == 8,也就是论文所说的
    d_ff = 2048
    num_blocks = 6
    num_heads = 8
    maxlen = 20
    dropout_rate = 0
    smoothing = 0.1
    vocab_fpath = "./data/vocab.txt"


