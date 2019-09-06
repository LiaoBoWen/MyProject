import tensorflow as tf
from tensorflow.contrib.layers import embed_sequence
import numpy as np


test = np.arange(4).reshape(2,2)

embeded = embed_sequence(test,10,embed_dim=10,initializer=tf.contrib.layers.xavier_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embeded))