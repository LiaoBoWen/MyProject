import tensorflow as tf
import os
import time
import numpy as np
import data_util
import similary
from BiLSTM import BiLSTM

embedding_file = 'zhwiki_2017_03.sg_50d.word2vec'
stop_word_file = 'data/stop_words.txt'
knowledge_file = 'data/knowledge.txt'
train_file = 'data/train.txt'
test_file = 'data/test.txt'
K = 5
max_sentence_len = 100
batch_size = 256
gpu_mem_usage = 0.75
embedding_dim = 50
rnn_size = 100
margin = 0.1 # todo 什么参数
learning_rate = 0.4
lr_down_times = 4
dropout_keep_drop = 0.45
num_epoches = 20
evaluate_every = 100
save_file = 'res/savedModel'
lr_down_rate = 0.5

if not os.path.exists('./res'):
    os.makedirs('./res')

# load pre_train embedding vector
print('loading embedding ...')
embedding, word2idx = data_util.load_embedding(embedding_file)
print(embedding)

# load stopwords
stop_words = open(stop_word_file,'r',encoding='utf8').readlines()
stop_words = [w.strip() for w in stop_words]

# top K most related knowlage
print('computing similarity....')
similary.generate_dic_and_corpus(knowledge_file,train_file,stop_words)
train_sim_ixs = similary.topK_sim_ix(train_file,stop_words,K)
test_sim_ixs = similary.topK_sim_ix(test_file,stop_words,K)

# Data preprocess begin
print('loading data...')
train_questions, train_answers, train_labels, train_question_num = data_util.load_data(knowledge_file,train_file,word2idx,stop_words,train_sim_ixs,max_sentence_len)
test_questions, test_answers, test_labels, test_questions_num = data_util.load_data(knowledge_file,test_file,word2idx,stop_words,test_sim_ixs,max_sentence_len)

print('检测：')
print(train_question_num,len(train_questions),len(train_answers),len(train_labels))
print(test_questions_num,len(test_questions),len(test_answers),len(test_labels))

questions, true_answers,  false_answers = [], [], []
for q, ta, fa in data_util.training_batch_iter(
    train_questions,train_answers,train_labels,train_question_num,batch_size):
    questions.append(q)
    true_answers.append(ta)
    false_answers.append(fa)

# Training begin
print('training ...')
with tf.Graph().as_default(), tf.device('/gpu:0'):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction = gpu_mem_usage
    )
    sess_config = tf.ConfigProto(
        allow_soft_placement = False,
        gpu_options=gpu_options
    )
    with tf.Session(config=sess_config).as_default() as sess:
        global_step = tf.Variable(0,name='global_step',trainable=False,)
        lstm = BiLSTM(
            batch_size,
            max_sentence_len,
            embedding,
            embedding_dim,
            rnn_size,
            margin
        )

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss,tvars),max_sentence_len)
        saver =  tf.train.Saver()

        timestamp = str(time.time())
        out_dir = os.path.abspath(os.path.join(os.path.curdir,'runs',timestamp))
        print('Write to {}\n'.format(out_dir))

        loss_summary = tf.summary.scalar('loss',lstm.loss)
        summary_op = tf.summary.merge([loss_summary])

        summary_dir = os.path.join(out_dir,'summary','train')
        summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)

        # evaluateing
        def evaluate():
            print('evaluating...')
            scores = []
            for test_q, test_a in data_util.testing_batch_iter(test_questions,test_answers, test_questions_num,batch_size):
                test_feed_dict = {
                    lstm.inputTestQuestions:test_q,
                    lstm.inputTestAnswers:test_a,
                    lstm.dropout_keep_out_prob:1.
                }
                _, score = sess.run([global_step,lstm.result],test_feed_dict)
                scores.append(score)
            cnt = 0
            scores = np.absolute(scores)
            for test_id in range(test_questions_num):
                offset = test_id * 4
                predict_true_ix = np.argmax(scores[offset:offset + 4])
                if test_labels[offset + predict_true_ix] == 1:
                    cnt += 1
            print('evaluation acc: {}'.format(cnt / test_questions_num))

            scores = []
            for train_q, train_a in data_util.testing_batch_iter(train_questions,train_answers,train_question_num,batch_size):
                test_feed_dict = {
                    lstm.inputTestQuestions:train_q,
                    lstm.inputTestQuestions:train_a,
                    lstm.dropout_keep_out_prob:1.
                }
                _, score = sess.run([global_step,lstm.result],test_feed_dict)
            cnt = 0
            scores = np.absolute(scores)
            for train_id in range(train_question_num):
                offset = train_id * 4
                predict_true_ix = np.argmax(scores[offset:offset + 4])
                if train_labels[offset + predict_true_ix] == 1:
                    cnt += 1
            print('evaluation acc(train): {}'.format(cnt / train_question_num))

        # training
        sess.run(tf.global_variables_initializer())
        learning_rate = learning_rate
        for i in range(lr_down_times):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # optimizer.apply_gradients(zip(grads,tvars))
            train_op = optimizer.apply_gradients(zip(grads,tvars),global_step=global_step)
            for epoch in range(num_epoches):
                for questions, trueAnswer, falseAnswer in zip(questions, true_answers, false_answers):
                    feed_dict = {
                        lstm.inputTestQuestions:questions,
                        lstm.inputTrueAnswer:trueAnswer,
                        lstm.inputFalseAnswer:falseAnswer,
                        lstm.dropout_keep_out_prob:dropout_keep_drop
                    }
                    _, step, _, _, loss, summary = sess.run([train_op,global_step,lstm.trueCosSim,lstm.falseCosSim,lstm.loss,summary_op],feed_dict)
                    print('ste: {} loss: {}'.format(step,loss))
                    summary_writer.add_summary(summary,step)
                    if step % evaluate_every == 0:
                        evaluate()

                saver.save(sess,save_file)
            learning_rate *= lr_down_rate
        # finaly evaluate
        evaluate()