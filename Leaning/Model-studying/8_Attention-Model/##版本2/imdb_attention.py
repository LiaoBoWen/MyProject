from keras.datasets import imdb
from keras.preprocessing import sequence

from attention import Position_Embedding, Attention

max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data')
(x_train,y_train), (x_test,y_test) = imdb.load_data(num_words=max_features)
print(x_train)
print(len(x_train),'train sequeces')
print(len(x_test),'test sequeces')

print('Pad sequence (samples * time)')
x_train = sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = sequence.pad_sequences(x_test,maxlen=maxlen)

print('x_train shape:',x_train.shape)
print('x_test shape:',x_test.shape)

from keras.models import Model
from keras.layers import *

S_inputs = Input(shape=(None,),dtype='int32')
embeddings = Embedding(max_features,128)(S_inputs)
embeddings = Position_Embedding()(embeddings) # 增加该层而可以适当的提升准确率
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1,activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs,outputs=outputs)
print(model.summary())


# 最好是不同的优化器进行优化
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Train...')
model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=5,
          # validation_data=(x_test,y_test)
          validation_split=.1
          )
## 报错了，重写的方法有问题：https://www.jianshu.com/p/380af57d8e9b


score, acc = model.evaluate(x_test,y_test,batch_size=batch_size)
print('Test score:',score)
print('Test accuracy:',acc)