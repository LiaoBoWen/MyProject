import tensorflow as tf

from data_preprocess import load_vocab
from modules import get_token_embeddings, feed_forward, position_encoding, multihead_attention,label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class Transformer:
    def __init__(self,hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab_path)
        self.embeddings = get_token_embeddings(self.hp.vocab_size,self.hp.num_units,zero_pad=True)

    def encode(self,xs,training=True):
        '''
        :return: encoder outputs (N,T1,hidden_units)
        '''
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # Embeding
            enc = tf.nn.embedding_lookup(self.embeddings.x)
            enc *= self.hp.num_units ** 0.5 # scale

            enc += position_encoding(enc,self.hp.maxlen)
            enc = tf.layers.dropout(enc,self.hp.dropout_rate,training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i),reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = feed_forward(enc,num_units=[self.hp.d_ff,self.hp.num_units])

        memory = enc
        return memory, sents1

    def decode(self,ys,memory,training=True):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # Embedding
            dec = tf.nn.embedding_lookup(self.embeddings,decoder_inputs)
            dec *= self.hp.num_units ** 0.5    # scale
            dec += position_encoding(dec,self.hp.maxlen)
            dec = tf.layers.dropout(dec,self.hp.dropout_rate,training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i),reuse=tf.AUTO_REUSE):
                    # Masked self-attention
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,   # todo 本次的causality是True
                                              scope='self_attention')
                    # attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,   # todo 此次的causality是False
                                              scope='vanilla_attention')
                    # Feed-Foward
                    dec = feed_forward(dec,num_units=[self.hp.d_ff,self.hp.num_units])

        # Final linear projection(embedding weights are shared)
        weights = tf.transpose(self.embeddings)  #[hidden_units,vocab_size]
        logits = tf.einsum('ntd,dk->ntk',dec,weights)  # (N,T2,vocab_size)
        # set values corresponding to unk = 0
        logits_first = tf.expand_dims(logits[:,:,0],2)
        zeros = tf.zeros_like(logits_first)
        logits = tf.concat([logits_first,zeros,logits[:,:,2:]],axis=2)
        y_hat = tf.to_int32(tf.argmax(logits,axis=-1))

        return logits,y_hat, y, sents2

    def train(self,xs,ys):
        # Forward
        memory, sents1 = self.encode(xs)
        logits, preds, y, sent2 = self.decode(ys,memory)

        # Train scheme
        y_ = label_smoothing(tf.one_hot(y,depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y,self.token2idx['<PAD>']))
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr,global_step,self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss,global_step=global_step)

        tf.summary.scalar('lr',lr)
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('global_step',global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step,summaries

    def infer(self,xs,ys):
        decoder_inputs, y, y_seqlen, sents2 = ys
        memory, _ = self.encode(xs,False)

        for _ in range(self.hp.maxlen):
            _, y_hat, y, y_sents2 = self.decode(ys,memory,False)
            if tf.reduce_sum(y_hat,1) == self.token2idx['<PAD>']: break

            _decoder_inputs = tf.concat((decoder_inputs,y_hat),1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        return y_hat

    def eval(self,xs,ys):
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0],1),tf.int32) * self.token2idx['<S>']
        ys = (decoder_inputs,y,y_seqlen,sents2)

        memory, sent1 = self.encode(xs,False)

        logging.info('Infernece graph is building, Please be patient.')
        for _ in tqdm(range(self.hp.maxlen)):
            logits, y_hat, y, sents2 = self.decode(ys,memory,False)
            if tf.reduce_sum(y_hat,1) == self.token2idx['<PAD>']:break

            _decoder_inputs = tf.concat((decoder_inputs,y_hat),1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((),0,tf.shape(y_hat)[0] - 1, tf.int32)
        sent1 = sents2[n]
        pred = convert_idx_to_token_tensor(y_hat[n],self.idx2token)
        sent2 = sents2[n]

        tf.summary.text('sent1',sent1)
        tf.summary.text('pred',pred)
        tf.summary.text('sent2',sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries
