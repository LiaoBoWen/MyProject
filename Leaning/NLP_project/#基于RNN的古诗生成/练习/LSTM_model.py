import tensorflow as tf
from tensorflow.contrib import rnn



class rnn_model:
    def __init__(self,num_word,input_data,output_data=None,hidden_size=128,num_layers=2,batch_size=128,cell_type='lstm',learning_rate = 1e-2):

        if cell_type == 'lstm':
            cell = rnn.BasicLSTMCell
        else:
            cell = rnn.GRUCell

        cell = cell(hidden_size)
        cell = rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)    #最后

        if output_data is not None:
            self.init_state = cell.zero_state(batch_size,tf.float32)
        else:
            self.init_state = cell.zero_state(1,tf.float32)

        embedding = tf.get_variable('embedding',initializer=tf.random_uniform([num_word + 1, hidden_size],-1.,1.))
        inputs = tf.nn.embedding_lookup(embedding,input_data)

        outputs, self.last_state = tf.nn.dynamic_rnn(cell,inputs,initial_state=self.init_state)
        self.output = tf.reshape(outputs,[-1,hidden_size])

        weight = tf.Variable(tf.truncated_normal([hidden_size,num_word + 1]))
        bias = tf.Variable(tf.zeros(shape=[num_word + 1]))
        logit = tf.nn.bias_add(tf.matmul(self.output,weight),bias)

        if output_data is not None:
            labels = tf.one_hot(tf.reshape(output_data,[-1]),depth=num_word + 1)    # 这里的depth类似于裁剪，所以没有毛病
            loss =  tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logit)
            self.total_loss = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.total_loss)
            tf.summary.scalar('loss',self.total_loss)

        else:
            self.prediction = tf.nn.softmax(logits=logit)