import tensorflow as tf
from config import config
from data_helper import load_json,padding

class Predict:
    def __init__(self,config,model_path='./runs/1548754630/checkpoints/model-1500',word2index='./vocabs/word2index.json',
                 index2label='./vocabs/index2label.json'):
        self.word2index = load_json(word2index)
        self.index2label = load_json(index2label)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config['allow_soft_placement'],
                log_device_placement = config['log_device_placement']
            )
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():    # 使用as_default(),当退出上下文的时候会话不会关闭
                # load model
                saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
                saver.restore(self.sess,model_path)

                # get the placeholders from graph by name
                self.input_x = graph.get_operation_by_name('input_x').outputs[0]

                self.dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

                # tensors we want to evaluate
                self.predictions = graph.get_operation_by_name('output/predictions').outputs[0]

    def predict(self,list_str):
        input_x = padding(list_str,None,config,self.word2index,None)
        feed_dict = {
            self.input_x:input_x,
            self.dropout_keep_prob:1.0
        }
        predictions = self.sess.run(self.predictions,feed_dict=feed_dict)
        return [self.index2label[str(idx)] for idx in predictions]

if __name__ == '__main__':
    prediction = Predict(config)
    result = prediction.predict(["黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 "])
    print(result)