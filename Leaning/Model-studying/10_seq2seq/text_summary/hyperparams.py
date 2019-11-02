class Params:
    vocab_size = 32000
    maxlen = 20
    epochs = 50
    batch_size = 32
    hidden_size = 200
    learning_rate = 0.0005
    keep_drop = 0.9

    print_per_batch = 20
    eval_per_batch = 50
    summary_file = './model_saved/summary'

    per_process_gpu_memory_fraction = 0.7
    processed_data_path = './data/data'
    idx2token_path = './model/idx2token.pkl'
    token2idx_path = './model/token2idx.pkl'
    model_save = './model_saved/chatbot/'
