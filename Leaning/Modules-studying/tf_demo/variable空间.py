import tensorflow as tf


with tf.name_scope('a'):
    with tf.name_scope('b'):
        test1 = tf.Variable(1.2,name='test1')
        test2 = tf.Variable(2.4,name='test2')
        test3 = tf.constant(0.0,name='test3')
        print(test3)
        print(test2)


with tf.variable_scope('room_1',reuse=tf.AUTO_REUSE):
    var_1 = tf.get_variable('var_1',initializer=[1,2],dtype=tf.int32)
    var_2 = tf.get_variable('var_1',initializer=[3,4],dtype=tf.int32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    test1_result, test2_result, test3_result, var_1_result, var_2_result = sess.run([test1,test2,test3,var_1,var_2])
    # sess.run(test2)
    print(test1.name, test2.name)
    print(var_1.name,var_2.name)
    print(sess.run([var_1,var_2]))
    print(test1_result, test2_result, test3_result,)

