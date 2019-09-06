# todo https://www.tensorflow.org/guide/datasets?hl=zh-cn
import tensorflow as tf
from tensorflow.contrib import data


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
print(dataset1.output_types)
print(dataset1.output_shapes)  # 这里要注意一下，由于第一个维度是batch_size，所以不计入shape

dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
     tf.random_uniform([4,100,2],maxval=100,dtype=tf.int32))
)

print(dataset2.output_types)
print(dataset2.output_shapes)

dataset3 = tf.data.Dataset.zip((dataset1,dataset2))
print(dataset3.output_types)
print(dataset3.output_shapes)

print('==================Dataset中的元素的各个组件======================')
# dataset1 = dataset1.map(lambda x:...)
#
# dataset2 = dataset2.flat_map(lambda x: ...)
#
# dataset3 = dataset3.filter(lambda x: ...)


print('===================单次迭代器======================')
dataset = tf.data.Dataset.range(10)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
result = tf.add(next_element,next_element)   # todo 这里需要注意，这里虽然有两个next_element但只run了一次next_element，原因：调用 Iterator.get_next() 并不会立即使迭代器进入下个状态。必须在 TensorFlow 表达式中使用此函数返回的 tf.Tensor 对象，并将该表达式的结果传递到 tf.Session.run()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        value = sess.run(result)
        print(value)

print('===============可重新初始化迭代器======================')
# todo 通过使用tf.data.Iterator构建迭代器
training_dataset = tf.data.Dataset.range(10).map(
    lambda x: x + tf.random_uniform([],-10,10,tf.int64)
)
validation_dataset = tf.data.Dataset.range(5)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# with tf.Session() as sess:
#     for _ in range(2):
#         sess.run(training_init_op)
#         for _ in range(10):
#             sess.run(next_element)
#
#         sess.run(validation_init_op)
#         for _ in range(5):
#             sess.run(next_element)


print('==================保存迭代器状态====================')

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

saveable = data.make_saveable_from_iterator(iterator)

tf.add_to_collection('test',saveable)

a = tf.Variable(13)

saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.save(sess,'./saved_iterator/r')
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess,'./saved_iterator/r')


print('=================repeat=====================')
dataset4 = tf.data.Dataset.range(4)
dataset4 = tf.data.Dataset.zip((dataset4,dataset4))

dataset4 = dataset4.repeat()
batched_dataset = dataset4.batch(2)
iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))


dataset = tf.data.Dataset.range(3)
dataset = dataset.map(lambda x: tf.expand_dims(tf.fill([2,2],x),axis=1))
dataset = dataset.repeat()
dataset = dataset.shuffle(1000)
dataset = dataset.padded_batch(2,padded_shapes=[None,None,None]).prefetch(1)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(next_element))



print('====================from_generator======================')
def test_generator():
    for i,j in zip(range(10),range(2,12)):
        yield [i] * 4, [j] * 4


output_shapes = ([None], [None])
output_types = (tf.int32, tf.int32)
output_pads = (100, 200)

dataset = tf.data.Dataset.from_generator(
    lambda : test_generator(),
    output_shapes = output_shapes,
    output_types= output_types
)

dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.repeat()
dataset = dataset.padded_batch(batch_size=4,padded_shapes=output_shapes,padding_values=output_pads).prefetch(1)
# dataset = dataset.make_one_shot_iterator()
# next_element = dataset.get_next()
iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types,output_shapes=dataset.output_shapes)

x, y = iterator.get_next()
init_op = iterator.make_initializer(dataset)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_op)
    for _ in range(3):
        print(sess.run(x))
        print(sess.run(y))
        print(sess.run([x,y]))