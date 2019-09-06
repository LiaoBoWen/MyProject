class params:
    vocab_size = 20000
    batch_size = 128
    eval_batch_size = 128
    learning_rate = 2e-5
    warmup_steps = 4000
    logdir = './model'
    num_epoch = 100
    evaldir = './model'
    num_units = 512
    d_ff = 2048
    num_blocks = 6   # encoder decoder 的块数
    num_heads = 8
    max_len = 20
    dropout_rate = 0
    smoothing = 0.1
    vocab_path = './data/vocab.txt'