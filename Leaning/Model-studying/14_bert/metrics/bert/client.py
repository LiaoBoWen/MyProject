from bert.run_classifier import *

class ExportModel:
    def __init__(self):
        self.processor = TextClsProcessor()

    def predict(self,inputs):

        inputs = [['0',input_] for input_ in inputs]
        predict_example = self.processor._create_examples(inputs, 'test')
        label_lst = self.processor.get_labels()

        tokenizor = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
                                               do_lower_case=FLAGS.do_lower_case)

        features = convert_examples_to_features(examples=predict_example,
                                                label_list=label_lst,
                                                max_seq_length=FLAGS.max_seq_length,
                                                tokenizer=tokenizor)

        predict_dataset = input_fn_builder(features=features,
                                           seq_length=FLAGS.max_seq_length,
                                           is_training=False,
                                           drop_remainder=False)()

        iterator = predict_dataset.make_one_shot_iterator()

        next_element = iterator.get_next()

        for input_ in ['label_ids','input_ids','input_masks','segment_ids']:
            next_element[input_] = next_element[input_].numpy().tolist()

        json_data = {'model_name':'default','data':next_element}

if __name__ == '__main__':
    model = ExportModel()
    model.predict(['感觉这个机子不怎么样','体验真的是才不舒服了，后悔了',''])