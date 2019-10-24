#### 模型流程
1. 修改之后的`InputExample`里面不包含text_b
2. 增加了对应的label；`InputFeature` 删除了判断是否是用一句
3. `DataProcessor`的文件读入进行修改:将分隔开来的字用空格进行连接（在`convert_single_example`可以按照空格切分得到句子序列）
4. 添加了`write_token`函数在test进行保存tokens
5. `convert_single_example`函数里面函数里面进行句子的序列化得到ntokens，id化，添加input_mask，label_id，segment_id, 同时将ntokens写入文件
6. ...数据处理、convert to feature
7. 创建模型，对比了有bilstm-crf层的模型，发现模型在test样本上的性能并没有比不添加bilstm_crf层的效果好，反而有略微的下降。
8. 添加bilstm_crf层的网络：crf_decode进行解码，viterbi_decode适用于单条数据，解码需要使用状态转移函数，所以训练的时使用的get_variable进行创建该变量，predict时时刻获取该变量
9. `label_ids`下标是从1开始，用下标0进行句子的padding，使的输入是固定长度
10. 验证时进行下标为2,3,4,5,6,7的进行验证，下标对应的是人物，地点，机构的标志label,使用的是macro方式
11. metrics函数调用[开源的metric代码](https://github.com/guillaumegenthial/tf_metrics)
12. 之后进行模型的提取，需要注意predict中传入的长度的类型需要是vector，list类型无法进行矢量减法会引起报错。