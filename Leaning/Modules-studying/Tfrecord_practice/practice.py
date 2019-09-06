# todo 从内存中数据构建
import tensorflow as tf
import os
import numpy as np
sess = tf.Session()

# labels = []
# features1 = []
# features2 = []
# with open('test.txt') as f:
#     for line in f:
#         splits = line.strip().split(' ')
#         label = int(splits[0])
#         fe1 = list(map(lambda x: float(x), splits[1].strip().split(',')))
#         fe2 = list(map(lambda x: x.strip(), splits[2].strip().split(',')))
#         labels.append(label)
#         features1.append(fe1)
#         features2.append(fe2)
# dataset = tf.data.Dataset.from_tensor_slices((labels,features1,features2))
# print(dataset.output_types)
# print(dataset.output_shapes)


# todo 从tfrecord格式的数据构建

print('==================1、生成tfrecord文件=================')

def _int64_feature(x):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

def _float_feature(x):
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))

def _bytes_feature(x):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))

def to_tfrecord(files_num=1000,batch_size=1000):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    # 由于tfrecord文件比较大，所以选择存储为多个文件时有必要的
    with open('./test.txt') as f:
        try:
            for i in range(files_num):
                with tf.python_io.TFRecordWriter('./data/data.tfrecords-{}-{}'.format(i,files_num)) as writer:
                    for j, line in enumerate(f):
                        if j == batch_size:
                            break
                        print('\r{}-{}'.format(i,j),end='')
                        splits = line.strip().split()
                        label = int(splits[0].strip())
                        fe1 = list(map(lambda x:float(x), splits[1].strip().split(',')))
                        fe2 = list(map(lambda x:x.strip().encode(), splits[2].strip().split(',')))

                        example = tf.train.Example(
                            features = tf.train.Features(
                                feature={
                                    'label':_int64_feature([label]),
                                    'feature1':_float_feature(fe1),
                                    'feature2':_bytes_feature(fe2),
                                }
                            )
                        )

                        writer.write(example.SerializeToString())
        except:
            return

# to_tfrecord()

print('==================2、创建dataset==================')           # todo  https://cloud.tencent.com/developer/article/1096597

def _parse_function(example_proto):
    features = {
        'label': tf.FixedLenFeature((),tf.int64),
        'feature1': tf.FixedLenFeature((5,),tf.float32),
        'feature2': tf.FixedLenFeature((2,),tf.string)
    }
    parsed_feature = tf.parse_single_example(example_proto,features)
    return parsed_feature['label'], parsed_feature['feature1'], parsed_feature['feature2']

def get_tfrecord_data():
    print('read...')
    dataset = tf.data.TFRecordDataset('data.tfrecords')
    print(dataset.output_types)
    print(dataset.output_shapes)

    dataset = dataset.map(_parse_function)
    print(dataset.output_types)
    print(dataset.output_shapes)


    dataset = dataset.shuffle(150)
    dataset = dataset.repeat()
    dataset = dataset.batch(125)

    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return next_element, init_op

def get_tfrecord_stream(epochs=200,shuffle=True):
    filenames = ['./data/{}'.format(x)  for x in  os.listdir('./data')]

    # 生成队列
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=epochs)

    reader = tf.TFRecordReader()

    # 返回文件名和文件   # 每次读取多个，这里指定10个
    _, serialized_example = reader.read_up_to(filename_queue,10)

    features = {
        'label': tf.FixedLenFeature([],tf.int64),
        'feature1': tf.FixedLenFeature([5],tf.float32),
        'feature2': tf.FixedLenFeature([2],tf.string),
    }

    # 同时解析所有样例
    parsed_feature = tf.parse_example(serialized_example,
                                             features)

    label = tf.cast(parsed_feature['label'],tf.int32)
    feature1 = tf.cast(parsed_feature['feature1'],tf.float32)
    feature2 = tf.cast(parsed_feature['feature2'],tf.string)


    batch_size = 64
    capacity = 100 + 3 * batch_size

    label.set_shape(10)
    feature1.set_shape([10,5])
    feature2.set_shape([10,2])


    if not shuffle:
        label_batch, feature1_batch, feature2_batch = tf.train.batch([label, feature1, feature2],batch_size=batch_size,
                                                                     capacity=capacity)

    # 不同线程处理各自的文件
    # 随机包含各个线程选择文件名的随机和文件内部数据读取的随机
    else:
        label_batch, feature1_batch, feature2_batch = tf.train.shuffle_batch([label, feature1, feature2], batch_size=batch_size,
                                                                     capacity=capacity,
                                                                     min_after_dequeue=64)  # 出队后队列中的最小数量元素，用于确保元素的混合程度。


    return label_batch, feature1_batch, feature2_batch




if __name__ == '__main__':
    next_element, init_op = get_tfrecord_data()
    sess.run(tf.global_variables_initializer())
    sess.run(init_op)

    tmp_count =  0
    while True:
        tmp_count += 1
        result = sess.run(next_element)
        print(tmp_count)


print('================使用为文件队列==============')
if __name__ == '__main__---------------':
    with tf.Session() as sess:
        label_batch, feature1_batch, feature2_batch = get_tfrecord_stream(epochs=500,shuffle=True)

        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

        # 使用num_epochs的话，需要在初始化全局变量和局部变量之前进行构建流，在初始化变量之后再进行接下来这两步
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):
            label_batch_, feature1_batch_, feature2_batch_ = sess.run([label_batch, feature1_batch, feature2_batch])
            # print(label_batch_)
            # print(feature1_batch_)
            # print(feature2_batch_)
            print('=' * 30,i)
        coord.request_stop()
        coord.join(threads)