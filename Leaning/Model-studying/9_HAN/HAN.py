import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib import rnn

class HAN:
    def __init__(self,accusation_num_classes,artical_num_classes,deathpenalty_num_classes,lifeimprisonment_num_classes,learning_rate,
                 batch_size,decay_steps,decay_rate,sequence_length,num_sentences,vocab_size,embed_size,hidden_size,is_training,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,max_pooling_style='max_pooling'):
        # Set all hyperparamenter here
        self.accusation_num_classes = accusation_num_classes
        self.article_num_classes = artical_num_classes
        self.deathpenalty_num_classes = deathpenalty_num_classes
        self.lifeimprisonment_num_classes = lifeimprisonment_num_classes
        self.batch_size = batch_size
        self.total_sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate,trainable=False,name='learning_rate')
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate,self.learning_rate * 0.5)
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.clip_gradlient = clip_gradients
        self.max_pooling_style = max_pooling_style

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32,[None,self.total_sequence_length],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,self.accusation_num_classes],name='input_y_accusation')
        self.input_y_accusation = tf.placeholder(tf.float32,[None,self.accusation_num_classes],name='input_y_accusation')
        self.input_y_article = tf.placeholder(tf.float32,[None,self.article_num_classes],name='input_y_article')
        self.input_y_deathpenalty = tf.placeholder(tf.float32,[None,self.deathpenalty_num_classes],name='input_y_deathpenalty')
        self.input_y_lifeimprisonment = tf.placeholder(tf.float32,[None,self.lifeimprisonment_num_classes],name='input_y_lifeimprisonment')
        self.input_y_imprisonment = tf.placeholder(tf.float32,[None],name='input_y_imprisonment')


        self.sequence_length = int(self.total_sequence_length / self.num_sentences)
        print('self.single_sequence_length:',self.sequence_length,';self.total_sequence_length:',self.total_sequence_length,';self.num_sentences:',self.num_sentences)
        self.dropout_keep_proba = tf.placeholder(tf.float32,name='dropout_keep_prob')
        self.global_step = tf.Variable(0,trainable=False,name='Global_Step')
        self.epoch_step = tf.Variable(0,trainable=False,name='Epoch_Step')
        self.epoch_increment =  tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps = self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits_accusation, self.logits_article,self.logits_deathpeanlty,self.logits_lifeimprisonment,self.logits_imprisonment = self.inference() # [None,self.label_size] main computation graph

        if not is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()

    def inference(self):
        # main computation graph here :1.Word Encoder 2.Word Attention 3. Sentence Encoder 4.Sentence Attention 5.Dropout 6.transform for each task 7.linear classifier

        # 1. Word Encoder: use bilstm to encoder the sentence.
        embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        embedded_words = [tf.squeeze(x) for x in tf.split(embedded_words,self.num_sentences,axis=1)]   # list.length is num_sentence,each element is ::[None,sentence_length,embed_size]
        word_attention_list = []
        for i in range(self.num_sentences):
            sentence = embedded_words[i]
            reuse_flag = True if i>0 else False
            # 2.word encoder
            num_units1 = self.embed_size
            sentence = tf.reshape(sentence,(-1,self.sequence_length,num_units1))
            word_encoded = self.bi_lstm(sentence,'word_level',num_units1,reuse_flag=reuse_flag)  # [batch_size,seq_length,numn_units * 2]
            # 3.word attention
            word_attention = self.attention(word_encoded,'word_level',reuse_flag=reuse_flag) # [batch_size, num_units * 2]
            word_attention_list.append(word_attention)
        sentence_encoder_input = tf.stack(word_attention_list,axis=1) # [batch_size,num_sentence,num_units * 2]
        # 4.sentence encoder
        num_units2 = self.hidden_size * 2
        sentence_encoder_input = tf.reshape(sentence_encoder_input,(-1,self.num_sentences,num_units2))
        sentence_encoded = self.bi_lstm(sentence_encoder_input,'sentence_level',num_units2) # [batch_size,seq_length,num_units * 4]
        # 5. sentence attention
        document_representation = self.attention(sentence_encoded,'sentence_level')  # [batch_size,num_units * 4]
        # 6. dropout
        h = tf.nn.dropout(document_representation,keep_prob=self.dropout_keep_proba) # [batch_size,num_units * 4]

        # 7. transform each sub task using one-layer MLP ,then get logits
        h_accusation = tf.layers.dense(h,self.hidden_size,activation=tf.nn.relu,use_bias=True)
        logits_accusation =  tf.layers.dense(h_accusation,self.accusation_num_classes,use_bias=True) # [None,slef.num_classes]

        h_article = tf.layers.dense(h,self.hidden_size,activation=tf.nn.relu,use_bias=True)
        logits_article = tf.layers.dense(h_article,self.article_num_classes,use_bias=True) #[None,self.num_classes]

        h_deathpenalty = tf.layers.dense(h,self.hidden_size,activation=tf.nn.relu,use_bias=True)
        logits_deathpenalty = tf.layers.dense(h_deathpenalty,self.deathpenalty_num_classes,use_bias=True) # [None,self.num_classes]

        h_lifeimprisonment = tf.layers.dense(h,self.hidden_size,activation=tf.nn.relu)
        logits_lifeimprisonment = tf.layers.dense(h_lifeimprisonment,self.lifeimprisonment_num_classes,use_bias=True) # [None,self.num_classes]

        print('logits_lifeimprisonment:',logits_lifeimprisonment)
        logits_imprisonment = tf.layers.dense(h,1,use_bias=True) # imprisonment 是一个延续的value，暴怒需要使用激活函数
        logits_imprisonment = tf.reshape(logits_imprisonment,[-1]) # [batch_size]

        return logits_accusation,logits_article,logits_deathpenalty,logits_lifeimprisonment,logits_imprisonment

    def attention(self,input_sequences,attention_level,reuse_flag=False):
        '''
        :param input_sequences: [batch_size,seq_length,num_units]
        :return:[batch_size,hidden_size]
        '''
        num_units = input_sequences.get_shape().as_list()[-1] # get last dimension
        with tf.variable_scope('attention_' + str(attention_level),reuse=reuse_flag):
            v_attention = tf.get_variable('u_attention' + attention_level, shape=[num_units],initializer=self.initializer)
            # 1.one-layer MLP
            u = tf.layers.dense(input_sequences,num_units,activation=tf.nn.tanh,use_bias=True) # [batch_size,seq_length,num_units]
            # 2.compute weight by compute simility of u and attention vector v
            score = tf.multiply(u,v_attention)  # [batch_size,seq_length,num_units]
            weight = tf.reduce_sum(score,axis=2,keepdims=True)  # todo keepdims
            # 3.weight sum
            attention_representation = tf.reduce_sum(tf.multiply(u,weight),axis=1) # [batch_size,num_units]
        return attention_representation

    def bi_lstm(self,input_sequences,level,num_units,reuse_flag=False):
        '''
        :param input_sequences:[batch_size,seq_length,num_units]
        :return: [batch_size,seq_length,num_units*2]
        '''
        with tf.variable_scope('bi-Lstm_' + str(level),reuse=reuse_flag):
            gru_fw_cell = rnn.GRUCell(num_units)
            gru_bw_cell = rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell,gru_bw_cell,input_sequences,dtype=tf.float32) # [batch_size,sequence_length]
        # concat output
        encoded_inputs = tf.concat(outputs,axis=2) # [batch_size,sequence_length,hidden_size *2]
        return encoded_inputs

    def loss(self,l2_lambda=1e-4):
        # input : `logits` and `labels` must have the same shape    [batch_size,num_classes]

        # loss1:accusation
        losses_accusation = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_accusation,logits=self.logits_accusation)
        loss_accusation = tf.reduce_mean((tf.reduce_sum(losses_accusation,axis=1))) # shape=(?,)-->(). loss for all data in the batch --> single loss

        # loss2:relevant article
        losses_article = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_article,logits=self.logits_article)
        loss_article = tf.reduce_mean((tf.reduce_sum(losses_article,axis=1)))

        # loss3:death penalty
        losses_deathpenalty = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_deathpenalty,logits=self.logits_deathpeanlty)
        loss_deathpenalty = tf.reduce_mean((tf.reduce_sum(losses_deathpenalty,axis=1)))

        # loss4:life imprisonment
        losses_lifeimprisonment = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_imprisonment,logits=self.logits_imprisonment)
        loss_lifeimprisonment = tf.reduce_mean((tf.reduce_sum(losses_lifeimprisonment,axis=1)))

        # loss5:imprisonment
        loss_imprisonment = tf.reduce_sum(tf.pow((self.logits_imprisonment - self.input_y_imprisonment),2)) / 50.0

        print('sigmoid_cross_entropy_with_logits.losses:',losses_accusation)    # shape:[?,1999]

        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        self.weights_accusation = tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 3.0
        self.weights_article = tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 3.0
        self.weights_deathpenalty = tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 3.0
        self.weights_lifeimprisonment = tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 3.0
        self.weights_imprisonment = 1 - tf.nn.sigmoid(tf.cast(self.global_step / 1000, dtype=tf.float32)) / 3.0

        loss = self.weights_accusation * loss_accusation + self.weights_article * loss_article + self.weights_deathpenalty * loss_deathpenalty + \
               self.weights_lifeimprisonment * loss_lifeimprisonment + self.weights_imprisonment * loss_imprisonment + l2_lambda * l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate,staircase=True)

        self.learning_rate_ = learning_rate
        train_op = tf_contrib.layers.optimize_loss(self.loss_val,global_step=self.global_step,
                                                   learning_rate=learning_rate,optimizer='Adam',clip_gradients=self.clip_gradlient
                                                   )
        return train_op

    def instantiate_weight(self):
        with tf.name_scope('embedding_project'):
            self.Embedding = tf.get_variable('Embedding',shape=[self.vocab_size,self.embed_size],initializer=self.initializer) # [vocab_size,embed_size] tf.random_uniform([self.vocab_size,self.embed_size]
