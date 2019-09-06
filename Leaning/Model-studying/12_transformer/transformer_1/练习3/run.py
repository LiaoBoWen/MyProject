from data_process import get_batch
from Model import Transformer
from hyperparams import Params
import tensorflow as tf

def train():
    session_conf = tf.ConfigProto(
        allow_soft_placement = True,
        log_device_placement = False
    )
    model_params = Params()
    model = Transformer(model_params)
    with tf.device('/gpu:0'):
        with tf.Session(config=session_conf) as sess:
            sess.run(tf.global_variables_initializer())

            for data,epoch_i in get_batch:
                xs, ys = data