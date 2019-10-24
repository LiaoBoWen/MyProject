from bert.run_classifier import convert_single_example
from bert.run_classifier import InputExample
from bert.run_classifier import InputFeatures
from bert.run_classifier import create_model
from bert import modeling
from bert import tokenization


import tensorflow as tf
from tensorflow.contrib.crf import crf_decode
import numpy as np
import pickle

flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('label2id_file', './ner_output/label2id.pkl','label2id path')
batch_size = 1
allow_soft_placement = True,




class Predict:
    def __init__(self):
        self.tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file)

        with open(FLAGS.label2id_file, 'rb') as f:
            self.label2id = pickle.load(f)
            self.id2label = {value: idx for idx, value in self.label2id.items()}


    def convert(self,line):
        feature = convert_single_example(0, line, self.label2id, FLAGS.max_seq_length, self.tokenizer, 'xxxx')
        input_ids = np.reshape(feature.input_ids, [batch_size, FLAGS.max_seq_length])
        input_mask = np.reshape(feature.input_mask, [batch_size, FLAGS.max_seq_length])
        segment_ids = np.reshape(feature.segment_ids, [batch_size, FLAGS.max_seq_length])
        laebls_ids = np.reshape(feature.label_ids, [batch_size, FLAGS.max_seq_length])
        return input_ids, input_mask, segment_ids, laebls_ids


    def convert_id_to_label(self, ids):
        ids = ids[0][1:]
        result = {'B-PER':[], 'B-LOC':[], 'B-ORG':[]}
        pointed_ids = {self.label2id['B-PER'], self.label2id['B-LOC'], self.label2id['B-ORG']}
        flag = False
        for i in range(len(ids)):
            if ids[i] == 1 or ids[i] == 9:
                continue
            if ids[i] == 10:
                break
            if ids[i] in pointed_ids:
                temp = [i]
                flag = self.id2label[ids[i]]
                continue
            if flag and ids[i + 1] != ids[i]:
                temp.append(i)
                result[flag].append(temp)
                flag = False

        return result


    def load_model(self,sess):
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)



        self.input_ids = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, [batch_size, FLAGS.max_seq_length], name='label_ids')
        # with tf.variable_scope('loss'):
        #     with tf.variable_scope('crf_loss'):
        #         transition = tf.get_variable('transitions', initializer=tf.zeros_initializer(), shape=[11, 11])

        # todo 这里的num_labels的长度要加1，因为还有PAD（0）
        self.loss, per_example_loss, self.logits, self.predict = create_model(bert_config, False,
                                                                   self.input_ids,
                                                                   self.input_mask,
                                                                   self.segment_ids,
                                                                   self.label_ids,
                                                                   len(self.label2id) + 1, False)

        checkpoint = tf.train.latest_checkpoint('ner_output/')
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        # self.pred, _ = crf_decode(potentials=self.input_logits, transition_params=trans, sequence_length=self.input_len)


    def run(self):
        # GPU_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            # gpu_options=GPU_options,
        )


        with tf.Session(config=sess_conf) as sess:
            self.load_model(sess)
            while True:
                raw = input('Input here:')
                # raw = '我叫廖博文'
                if not raw.strip():
                    return 'End'
                length = len(raw)
                input_sentence = ' '.join(raw.split())
                labels = ' '.join(['O' for _ in range(length)])
                input_sentence = InputExample(guid=0,text=input_sentence,label=labels)
                input_ids, input_mask, segment_ids, labels_ids = self.convert(input_sentence)
                feed_dict = {self.input_ids:input_ids,
                             self.input_mask:input_mask,
                             self.segment_ids:segment_ids,
                             self.label_ids:labels_ids,
                             }
                preds = sess.run(self.predict, feed_dict=feed_dict)

                # feed_dict = {self.input_logits:logits,
                #              self.input_len:np.array([len(raw)])}
                # preds = sess.run(self.pred, feed_dict=feed_dict)
                # print(preds)

                result_id = self.convert_id_to_label(preds)

                PERs, LOCs, ORGs,  = [], [], []
                for pair in result_id['B-PER']:
                    PERs.append(raw[pair[0]:pair[1] + 1])
                for pair in result_id['B-LOC']:
                    LOCs.append(raw[pair[0]:pair[1] + 1])
                for pair in result_id['B-ORG']:
                    ORGs.append(raw[pair[0]:pair[1] + 1])

                print('PER:', PERs)
                print('LOC:', LOCs)
                print('ORG:', ORGs)
                # return PERs, LOCs, ORGs


if __name__ == '__main__':
    predictor = Predict()
    predictor.run()