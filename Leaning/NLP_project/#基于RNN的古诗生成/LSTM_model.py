import tensorflow as tf
from tensorflow.contrib import rnn

learning_rate = 1e-2

def rnn_model(num_of_word,input_data,output_data=None,rnn_size=128,num_layers=2,batch_size=128,cell_type="lstm"):
    '''
    :param num_of_word: 单词个数
    :param input_data: 输入向量
    :param output_data: 标签
    :param rnn_size: 隐藏层的向量尺寸
    :param num_layers: 隐藏层的层数
    :param batch_size:
    :return:
    '''
    end_points = {}

    # 构建RNN核心
    if cell_type.lower() == 'gru':
        cell_fun = rnn.GRUCell
    else:
        cell_fun = rnn.BasicLSTMCell

    cell = cell_fun(rnn_size,state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell for _ in range(num_layers)] , state_is_tuple=True)

    # todo 如果有标签(output_data)则初始化一个batch的cell状态，否则初始化一个 : 预测的时候加一个
    if output_data is not None:
        initial_state = cell.zero_state(batch_size,tf.float32)
    else:
        initial_state = cell.zero_state(1,tf.float32)

    # 词嵌入核心
    embedding = tf.get_variable('embedding',initializer=tf.random_uniform([num_of_word + 1,rnn_size],-1.,1.))
    inputs = tf.nn.embedding_lookup(embedding,input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
    output = tf.reshape(outputs,[-1,rnn_size])

    # todo 这里的维度有点魔幻……
    weights = tf.Variable(tf.truncated_normal([rnn_size,num_of_word + 1]))
    bias = tf.Variable(tf.zeros(shape=[num_of_word + 1]))
    logits = tf.nn.bias_add(tf.matmul(output,weights),bias)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data,[-1]),depth=num_of_word + 1)   # 在这里的reshape起到了去除括号的作用
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
        tf.summary.scalar('loss',total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points