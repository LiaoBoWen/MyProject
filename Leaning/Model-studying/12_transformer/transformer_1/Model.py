import tensorflow as tf

from Data_process import load_vocab
from Modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
from Hyperparams import hyperparams as hp
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab_fpath)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.num_units, zero_pad=True)    # todo 这里考虑到PAD的存在，所以增加了0这一行

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, hidden_units)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, hidden_units)
            enc *= self.hp.num_units**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.num_units])
        memory = enc
        return memory, sents1

    def decode(self, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, hidden_units)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, hidden_units)
            dec *= self.hp.num_units ** 0.5  # scale

            dec += positional_encoding(dec, self.hp.maxlen)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.num_units])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (hidden_units, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        # set values corresponding to unk = 0    todo force index[1] == 0
        logits_first = tf.expand_dims(logits[:,:,0], 2)
        zeros = tf.zeros_like(logits_first)
        logits = tf.concat([logits_first, zeros, logits[:,:,2:]], axis=2)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, sents1 = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory)
        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<PAD>"]))  # 0: <PAD>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)  # 把PAD的无意义值去除

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def infer(self, xs, ys):
        decoder_inputs, y, y_seqlen, sents2 = ys
        memory, _ = self.encode(xs, False)

        for _ in range(self.hp.maxlen):
            _, y_hat, y, sents2 = self.decode(ys, memory, False)                # todo 这里的yhat预测出来之后还有再次进入decoder，是为了进行纠错？
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<PAD>"]: break    # 这里的1维度就是倒数第二维度

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        return y_hat

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<S>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<PAD>"]: break   # 当所有的句子都预测为pad，就结束预测

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample   todo what

        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries
