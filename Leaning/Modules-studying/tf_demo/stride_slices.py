import numpy as np
import tensorflow as tf


test = np.arange(8).reshape(2,4)
print(test)

demo = tf.strided_slice(test,[0,0],[2,-2],[1,1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(demo))