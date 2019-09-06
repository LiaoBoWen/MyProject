from tqdm import tqdm
import numpy as np

import tensorflow as tf

from Modules import get_token_embedding, position_encode, multi_attention, feed_forward, label_smoothing, noam_scheme
from utils import *
from Data_process import get_vocab

class Transformer:
    def __init__(self,hp):
        self.hp = hp
        self.token2idx,self.idx2token = get_vocab(self.hp.vocab_path)
        self.embedding = get_token_embedding

    def encoder(self,xs,training=True):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            x, seqlen, sents1 = xs

            enc = tf.nn.embedding_lookup(self.embedding,x)
            enc *= self.hp.num_unit ** 0.5

            enc += position_encode(enc,self.hp.vocab_size)
            enc = tf.layers.dropout(enc,self.hp.dropout_rate,training=True)

            enc = multi_attention(queries=enc,
                                  keys=enc,
                                  values=enc,
                                  causality=False,
                                  num_heads=self.hp.num_heads,
                                  training=training,
                                  dropout=self.hp.dropout_rate)

            enc = feed_forward(enc,num_units=[self.hp.d_ff,self.hp.num_unit])

        return enc, sents1

    def decoder(self,ys,memory,training=True):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            decode_input, y, seqlen, sents2 = ys

            dec = tf.nn.embedding_lookup(self.embedding,decode_input)
            dec *= self.hp.num_unit ** 0.5

            dec += position_encode(dec, self.hp.num_unit)
            dec = tf.layers.dropout(dec,self.hp.dropout_rate,training=training)

            dec = multi_attention(queries=dec,
                                  keys=dec,
                                  values=dec,
                                  causality=False,
                                  training=training,
                                  num_heads=self.hp.num_heads,
                                  dropout=self.hp.dropout_rate)

            dec = multi_attention(queries=dec,
                                  keys=memory,
                                  values=memory,
                                  causality=True,
                                  dropout=self.hp.dropout_rate,
                                  num_heads=self.hp.num_heads,
                                  training=training)

            dec = feed_forward(dec,num_units=[self.hp.d_ff,self.hp.num_unit])

        weights =  tf.transpose(dec)
        logits = tf.einsum('ntk,km->ntm',dec, weights)
        logit_first = tf.expand_dims(logits[:,:,0],-1)
        zero_ = tf.zeros_like(logit_first)
        logits = tf.concat((logit_first, zero_,logits[:,:,2]),-1)
        yhat = tf.reduce_max(logits,axis=-1)

        return logits, yhat, y, sents2

    def train(self,xs,ys):
        memory, sents1 = self.encoder(xs,training=True)
        logits, yhat, y, sents2 = self.decoder(ys,memory,training=True)

        y_ = label_smoothing(tf.one_hot(y,depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=logits)
        nonpadding = tf.to_float(tf.not_equal(ce,self.token2idx['<PAD>']))
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.learning_rate,global_step,self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

        tf.summary.scalar('lr',lr)
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('global_step',global_step)

        summaries = tf.summary.merge_all()

        return train_op, loss, global_step, summaries


    def eval(self,xs,ys):
        decode_input, y, seqlen,sents2 = ys
        memory, sents1 = self.encoder(xs,training=False)

        for _ in range(self.hp.maxlen):
            logits, yhat, y, sents2 = self.decoder(ys,memory,training=False)
            if tf.reduce_sum(yhat,1) == self.token2idx['<PAD>']:
                break
            decode_input = tf.concat((decode_input,yhat),1)
            ys = (decode_input, y, seqlen, sents2)

        n = tf.random_uniform((),0,self.hp.maxlen,tf.int32)
        sent1 = sents1[n]
        sent2 = sents2[n]
        pred = convert_to_tensor(yhat)

        tf.summary.text('sent1',sent1)
        tf.summary.text('sent2',sent2)
        tf.summary.text('pred',pred)

        summaries = tf.summary.merge_all()
        return yhat, summaries



    def infer(self,xs,ys):
        decode_input, y, seqlen, sents2 = ys

        memory, sents1 = self.encoder(xs,training=False)

        for _ in range(self.hp.maxlen):
            logits, yhat, y, sents2 = self.decoder(ys,memory,training=False)
            if tf.reduce_sum(yhat,1) == self.token2idx['<PAD>']:
                break
            decode_input = tf.concat((decode_input,yhat),1)
            ys = (decode_input, y, seqlen, sents2)

        return yhat
