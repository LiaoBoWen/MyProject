# '''
# estimator 原生并不支持一次加载，多次预测。参见：https://guillaumegenthial.github.io/serving-tensorflow-estimator.html。
# 因此需要使用 estimator.export_saved_model() 方法把 estimator 重新导出成 tf.saved_model。
#
# 以分类为例
# '''
# import tensorflow as tf
#
# flags = tf.flags
# FLAGS = flags.FLAGS
#
# def serving_input_fn():
#     label_ids = tf.placeholder(tf.int32, [None], name='lable_ids')
#     input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_len], name='input_ids')
#     input_mask = tf.placeholder(tf.int32, [None,FLAGS.max_seq_len], name='input_mask' )
#     sengment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_se_len], name='segment_ids')
#     input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
#         {'label_ids': label_ids,
#          'input_ids': input_ids,
#          'input_mask': input_mask,
#          'segment_ids':sengment_ids}
#     )()
#     return input_fn
#
# estimator._export_to_tpu = False
# # 会有一个时间戳命名的模型目录，因此最终模型是my_model/1523421132
# estimator.export_savedmodel('my_model',serving_input_fn)
#
# predict_fn = tf.contrib.predictor.from_saved_model('my_model/1523421132')
#
# while True:
#     question = input()
#     predict_fn = convert_single_example(100, predict_example, label_list,
#                                         FLAGS.max_seq_len, tokenizer)
#
#     prediction = predict_fn({
#         'input_ids':[feature.input_ids],
#         'input_mask':[feature.input_mask],
#         'segment_ids':[feature.segment_ids],
#         'label_ids':[feature.label_id]
#     })
#     prob = prediction['probabilities']
#     label = label_list[prob.argmax()]
#     print(label)


import tensorflow as tf
from bert import modeling
from bert import tokenization
import numpy as np
import pandas as pd

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_single_example(max_seq_length,tokenizer,text_a, text_b=None):
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)

  if tokens_b:
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  return input_ids, input_mask, segment_ids

bert_config = modeling.BertConfig.from_json_file('/media/liao/Data/temp_data/chinese_L-12_H-768_A-12/bert_config.json')
vocab_file = '/media/liao/Data/temp_data/chinese_L-12_H-768_A-12/vocab.txt'
batch_size = 20
num_labels = 3
is_training = True
max_seq_length = 128
iter_num = 20
lr = 0.00005
if max_seq_length > bert_config.max_position_embeddings:
    raise ValueError('over max_len')

with open('data/train.tsv', encoding='utf-8') as f:
    data = pd.read_csv(f, sep='\t')
    data['label'] = data['label'].astype(np.int8)
    texts = data['txt'].values
    labels = data['label'].values

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file) # 用于分字， 转换成id
input_idsList = []
input_masksList = []
segment_idsList = []

for t in texts:
    single_input_id, single_input_mask, single_segment_id = convert_single_example(max_seq_length, tokenizer, t)
    input_idsList.append(single_input_id)
    input_masksList.append(single_input_mask)
    segment_idsList.append(single_segment_id)

input_idsList = np.asarray(input_idsList, dtype=np.int32)
input_masksList = np.asarray(input_masksList, dtype=np.int32)
segment_idsList = np.asarray(segment_idsList, dtype=np.int32)


input_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length], name='input_ids')
input_masks = tf.placeholder(tf.int32, [batch_size, max_seq_length], name='input_masks')
segment_ids = tf.placeholder(tf.int32, [batch_size, max_seq_length], name='segment_ids')

input_labels = tf.placeholder(tf.int32, [batch_size], name='input_ids')

model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids = input_ids,
    input_mask = input_masks,
    token_type_ids = segment_ids,
    use_one_hot_embeddings = False
)

output_layer = model.get_sequence_output()
output_layer = model.get_pooled_output()
hidden_size = output_layer.shape[-1].value

output_weights = tf.get_variable(
    'output_weights', [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02)
)

output_bias = tf.get_variable('output_bias', [num_labels], initializer=tf.zeros_initializer())

with tf.variable_scope('loss'):
    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob= 0.9)
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(input_labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    predict = tf.argmax(tf.nn.softmax(logits), axis=1, name='predictions')
    acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, tf.cast(predict, dtype=tf.int32)), 'float'), name='accuracy')

train_op = tf.train.AdamOptimizer(lr).minimize(loss)

init_checkpoint = '/media/liao/Data/temp_data/chinese_L-12_H-768_A-12/bert_model.ckpt'
use_tpu = False

tvars = tf.trainable_variables()

(assignment_map, initialized_varibale_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

tf.logging.info('***trainable variable ***')

for var in tvars:
    init_string = ''
    if var.name in initialized_varibale_names:
        init_string = '*INIT_FROM_CKPT*'
    tf.logging.info('name={}, shape={}{}'.format(var.name, var.shape, init_string))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_nums = len(texts) // batch_size
    for i in range(iter_num):
        shuffleindex = np.random.permutation(np.arange(len(texts)))
        for j in range(batch_nums):
            batch_labels = labels[shuffleindex[j * batch_size: j * batch_size + batch_size]]
            batch_input_idsList = input_masksList[shuffleindex[j * batch_size: j * batch_size + batch_size]]
            batch_input_masksList = input_masksList[shuffleindex[j * batch_size: j * batch_size + batch_size]]
            batch_segment_idsList = segment_idsList[shuffleindex[j * batch_size: j * batch_size + batch_size]]
            l, a, _ = sess.run([loss, acc, train_op], feed_dict={
                input_ids:batch_input_idsList,
                input_masks:batch_input_masksList,
                segment_ids:batch_segment_idsList,
                input_labels:batch_labels
            })
            if j % 20 == 0:
                print('acc:{} , loss:{}'.format(a, l))

# 精度低， 不稳定，在0.6,0.85之间徘徊