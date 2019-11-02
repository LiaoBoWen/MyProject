import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.layers import xavier_initializer
import os
from data_processor import load_processed_data
from data_processor import load_vocab
from data_processor import get_batch
from hyperparams import Params


class Model:
    def __init__(self, vocab_to_int, batch_size, hidden_dim, learning_rate, vocab_size):
        # self.batch_size = batch_size
        self.vocab_to_int = vocab_to_int
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.temp_loss = tf.Variable(1000., trainable=False,name='temp_loss', dtype=tf.float32)

        self.add_placeholder()
        self.train()

    def add_placeholder(self):
        self.input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.text_length = tf.placeholder(tf.int32, [None, ], name='text_length')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
        self.summary_length = tf.placeholder(tf.int32, [None, ], name='summary_length')
        self.batch_size = tf.shape(self.input_data)[0]
        self.max_summary_length = tf.reduce_max(self.summary_length, name='max_summary')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.embeddings = tf.get_variable('embeddings',
                                          [self.vocab_size, 200],
                                          initializer=tf.random_uniform_initializer(-1., 1.),
                                          dtype=tf.float32)

    def process_encoding_input(self, target_data):
        # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
        ending = tf.strided_slice(target_data, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab_to_int['<S>']), ending], axis=1)

        return dec_input

    def encoding_layer(self, sequence_length, rnn_input, keep_prob):
        with tf.variable_scope('encode'):
            cell_fw = rnn.LSTMCell(self.hidden_dim, initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2))
            cell_fw = rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = rnn.LSTMCell(self.hidden_dim, initializer=tf.random_normal_initializer(-0.1, 0.1, seed=2))
            cell_bw = rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            enc_out, (enc_state_f, enc_state_b) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_input,
                                                                                  sequence_length, dtype=tf.float32)

        enc_out = tf.concat(enc_out, axis=2)
        enc_state = (tf.concat([enc_state_f[0], enc_state_b[0]], axis=1),
                     tf.concat([enc_state_f[1], enc_state_b[1]], axis=1))

        return enc_out, enc_state

    def training_decoding_layer(self, dec_embed_input, summary_len, dec_cell, initial_state, output_layer,
                                max_summary_len):
        training_helper = seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                 sequence_length=summary_len,
                                                 time_major=False)
        training_decoder = seq2seq.BasicDecoder(dec_cell, training_helper,
                                                initial_state, output_layer)
        training_logits, _, _ = seq2seq.dynamic_decode(training_decoder, output_time_major=False,
                                                       impute_finished=True,
                                                       maximum_iterations=max_summary_len)

        return training_logits

    def inference_decoding_layer(self, embeddings, start_token, end_token, dec_cell,
                                 initial_state, output_layer, max_summary_len):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size], name='start_token')

        inference_helper = seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens, end_token)

        inference_decoder = seq2seq.BasicDecoder(dec_cell,
                                                 inference_helper,
                                                 initial_state,
                                                 output_layer)

        inference_logits, _, _ = seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                        maximum_iterations=max_summary_len)

        return inference_logits

    def decoding_layer(self, dec_embed_input, embeddings, enc_output, enc_state,
                       vocab_size, text_len, summary_len, max_sum_len):
        lstm = rnn.LSTMCell(self.hidden_dim * 2, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

        dec_cell = rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_prob, )

        output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        attn_mech = seq2seq.BahdanauAttention(self.hidden_dim * 2,
                                              enc_output,
                                              text_len,
                                              normalize=False, name='BahdanauAttention')

        dec_cell = seq2seq.AttentionWrapper(dec_cell, attn_mech, attention_layer_size=self.hidden_dim * 2)

        # initial_state = seq2seq.AttentionWrapperState(enc_state[0],_zero_state_tensors(self.hidden_dim,batch_size,
        #                                                                                tf.float32))
        initial_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=LSTMStateTuple(*enc_state))

        with tf.variable_scope('decode'):
            traing_logits = self.training_decoding_layer(dec_embed_input, summary_len, dec_cell, initial_state,
                                                         output_layer, max_sum_len)

        with tf.variable_scope('decode', reuse=True):
            inference_logits = self.inference_decoding_layer(embeddings, self.vocab_to_int['<S>'],
                                                             self.vocab_to_int['</S>'], dec_cell,
                                                             initial_state, output_layer, max_sum_len)

        return traing_logits, inference_logits

    def seq2seq_model(self, text_len, target_data, summary_len, max_sum_len, vocab_size):
        enc_embed_input = tf.nn.embedding_lookup(self.embeddings, self.input_data)
        enc_output, enc_state = self.encoding_layer(text_len, enc_embed_input, self.keep_prob)

        dec_input = self.process_encoding_input(target_data=target_data)
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings, dec_input)

        training_logits, inference_logits = self.decoding_layer(dec_embed_input, self.embeddings,
                                                                enc_output, enc_state, vocab_size,
                                                                text_len, summary_len, max_sum_len)

        return training_logits, inference_logits

    def train(self):
        self.global_step = tf.train.get_or_create_global_step()
        training_logit, inference_logit = self.seq2seq_model \
            (self.text_length, self.targets, self.summary_length, self.max_summary_length, self.vocab_size)

        # create tensor for train_logit and inference_logit
        self.training_logits = tf.identity(training_logit.rnn_output, 'logits')
        self.inference_logits = tf.identity(inference_logit.sample_id, 'predictions')

        # create weights for sequence_loss
        masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32, name='masks')

        with tf.variable_scope('optimization'):
            self.cost = seq2seq.sequence_loss(self.training_logits,
                                              self.targets,
                                              masks)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = optimizer.compute_gradients(self.cost)
            cliped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(cliped_gradients, global_step=self.global_step)

        tf.summary.scalar('loss', self.cost)
        self.summary = tf.summary.merge_all()


class another_seq2seq:
    def __init__(self, vocab2ids, batch_size, hidden_dim, learning_rate, beam_width, embedding_size):
        self.vocab2ids = vocab2ids
        self.vocab_size = len(vocab2ids)
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.Beam_width = beam_width

        self.add_placeholders()
        self.train()

    def add_placeholders(self):
        self.source = tf.placeholder(tf.int32, [None, None], name='source')
        self.target = tf.placeholder(tf.int32, [None, None], name='target')
        self.source_len = tf.placeholder(tf.int32, [None, ], name='source_len')
        self.target_len = tf.placeholder(tf.int32, [None, ], name='target_len')
        self.max_target_len = tf.reduce_max(self.target_len)
        self.keepProb = tf.placeholder(tf.float32, name='keep_prob')

        self.embeddings = tf.get_variable('embeddings', shape=[self.vocab_size, self.embedding_size],
                                          initializer=xavier_initializer(seed=32))

    def encoder(self, enc_input, seq_len, keep_prob):
        cell_fw = rnn.LSTMCell(self.hidden_dim)
        cell_bw = rnn.LSTMCell(self.hidden_dim)
        cell_fw = [rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)]
        cell_bw = [rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)]
        enc_output, enc_state_f, enc_state_b = rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, enc_input,
                                                                                   sequence_length=seq_len,
                                                                                   dtype=tf.float32)

        self.enc_output = tf.concat(enc_output, axis=2)
        self.enc_state = rnn.LSTMStateTuple(c=tf.concat([enc_state_f[0].c, enc_state_b[0].c], axis=1),
                                            h=tf.concat([enc_state_f[0].h, enc_state_b[0].h], axis=1))

    def process_decode_input(self, enc_output):
        following = tf.strided_slice(enc_output, [0, 0], [self.batch_size, -1], [1, 1])
        # 这里可以取出following的shape来获取batch_size, 就可以来应对batch_size不同大小的data
        processed_encode = tf.concat([tf.fill([self.batch_size, 1], self.vocab2ids['<S>']), following], axis=1)

        return processed_encode

    def train_decode_layer(self, dec_embeddig_input, dec_cell, output_layer):
        atten_mech = seq2seq.BahdanauAttention(num_units=self.hidden_dim * 2,
                                               memory=self.enc_output,
                                               memory_sequence_length=self.target_len,
                                               normalize=True,
                                               name='BahadanauAttention'
                                               )
        dec_cell = seq2seq.AttentionWrapper(dec_cell, atten_mech, self.hidden_dim * 2, name='dec_attention_cell')

        initial_state = dec_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
            cell_state=self.enc_state)

        train_helper = seq2seq.TrainingHelper(dec_embeddig_input, self.target_len)
        training_decoder = seq2seq.BasicDecoder(dec_cell, train_helper, initial_state=initial_state,
                                                output_layer=output_layer)
        train_logits, _, _ = seq2seq.dynamic_decode(training_decoder,
                                                    output_time_major=False,
                                                    impute_finished=False,
                                                    maximum_iterations=self.max_target_len)
        return train_logits

    def inference_decode_layer(self, start_token, dec_cell, end_token, output_layer):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size], name='start_token')
        tiled_enc_output = seq2seq.tile_batch(self.enc_output, multiplier=self.Beam_width)
        tiled_enc_state = seq2seq.tile_batch(self.enc_state, multiplier=self.Beam_width)
        tiled_source_len = seq2seq.tile_batch(self.source_len, multiplier=self.Beam_width)
        atten_mech = seq2seq.BahdanauAttention(self.hidden_dim * 2,
                                               tiled_enc_output,
                                               tiled_source_len,
                                               normalize=True)
        decoder_att = seq2seq.AttentionWrapper(dec_cell, atten_mech, self.hidden_dim * 2)
        initial_state = decoder_att.zero_state(self.batch_size * self.Beam_width, tf.float32).clone(
            cell_state=tiled_enc_state)
        decoder = seq2seq.BeamSearchDecoder(decoder_att,
                                            self.embeddings,
                                            start_tokens,
                                            end_token,
                                            initial_state,
                                            beam_width=self.Beam_width,
                                            output_layer=output_layer)
        infer_logits, _, _ = seq2seq.dynamic_decode(decoder,
                                                    False, False,
                                                    self.max_target_len)
        return infer_logits

    def decoder(self, dec_embed_input, vocab_size):
        dec_cell = rnn.LSTMCell(self.hidden_dim * 2)
        dec_cell = rnn.DropoutWrapper(dec_cell, input_keep_prob=self.keepProb)
        output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.random_normal_initializer(-0.1, 0.1, seed=32))

        with tf.variable_scope('train_decoder'):
            train_logit = self.train_decode_layer(dec_embed_input, dec_cell, output_layer)
        with tf.variable_scope('inference_decode'):
            inference_logit = self.inference_decode_layer(self.vocab2ids['<S>'],
                                                          dec_cell,
                                                          self.vocab2ids['</S>'],
                                                          output_layer)
        return train_logit, inference_logit

    def seq2seq(self):
        embeded_input = tf.nn.embedding_lookup(self.embeddings, self.source)
        self.encoder(embeded_input, self.source_len, self.keepProb)

        dec_input = self.process_decode_input(self.target)
        dec_embed_input = tf.nn.embedding_lookup(self.embeddings, dec_input)
        train_logit, infer_logit = self.decoder(dec_embed_input, self.vocab_size)
        self.train_logit = tf.identity(train_logit.rnn_output, name='train_logit')
        self.infer_logit = tf.identity(infer_logit.predicted_ids, name='infer_logit')

    def train(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.seq2seq()

        mask = tf.sequence_mask(self.target_len, self.max_target_len, dtype=tf.float32, name='mask')
        with tf.variable_scope('optimizer'):
            self.cost = seq2seq.sequence_loss(self.train_logit, self.target, weights=mask)

            opetimizer = tf.train.AdamOptimizer(self.learning_rate)
            graidents = opetimizer.compute_gradients(self.cost)
            cliped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in graidents if grad is not None]
            self.train_op = opetimizer.apply_gradients(cliped_gradients)




save_path = Params.model_save

if not os.path.exists('./model_saved'):
    os.makedirs('./model_saved')
if not os.path.exists(save_path):
    os.makedirs(save_path)

id2token, token2id = load_vocab()
train_data, test_data = load_processed_data()


def train():

    with tf.Session() as sess:

        # model = another_seq2seq(vocab_to_int,batch_size,hidden_size,learning_rate,3,200)

        model = Model(token2id, Params.batch_size, Params.hidden_size, Params.learning_rate, len(token2id))
        summary_filewrite = tf.summary.FileWriter(Params.summary_file, sess.graph)

        saver = tf.train.Saver(max_to_keep=3)

        checkpoint = tf.train.latest_checkpoint(save_path)
        if checkpoint:
            saver.restore(sess, checkpoint)
            global_step, temp_loss = sess.run([model.global_step, model.temp_loss])
            print('Restore model from {}. global_step:{}, saved_loss:{}'.format(checkpoint, global_step, temp_loss))
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialed model.')

        for epoch_i in range(Params.epochs):

            for source_batch, target_batch, source_len, target_len, batch_i in get_batch(train_data, epoch=50, batch_size=32, shuffle=True):
                _, loss, predicts, summary_, global_step = sess.run([model.train_op, model.cost, model.inference_logits, model.summary, model.global_step],
                                             feed_dict={model.input_data: source_batch,
                                                        model.targets: target_batch,
                                                        model.text_length: source_len,
                                                        model.summary_length: target_len,
                                                        model.keep_prob: Params.keep_drop})
                summary_filewrite.add_summary(summary_, global_step)


                if global_step % Params.eval_per_batch == 0:
                    temp_loss = 0
                    for source_batch, target_batch, source_len, target_len, batch_i in get_batch(test_data, epoch=1,
                                                                                                 batch_size=32,
                                                                                                 shuffle=False):
                        eval_loss, predicts = sess.run(
                            [model.cost, model.inference_logits],
                            feed_dict={model.input_data: source_batch,
                                       model.targets: target_batch,
                                       model.text_length: source_len,
                                       model.summary_length: target_len,
                                       model.keep_prob: 1.})
                        temp_loss += eval_loss
                    saved_loss = sess.run(model.temp_loss)
                    if temp_loss < saved_loss:
                        sess.run(tf.assign(model.temp_loss, temp_loss))
                        saver.save(sess, save_path + 'checkpoint', model.global_step)
                        print('Saved model with lower loss:{}\n'.format(temp_loss))

                if global_step % Params.print_per_batch == 0:
                    print('Global_step:{} Epoch:{} loss:{}'.format(global_step, epoch_i + 1, loss))


if __name__ == '__main__':
    train()