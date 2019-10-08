# 1.can extract embeddings
# 2.how to load the model
#

# bert-server
from bert_serving.client import BertClient
bc = BertClient()
result1 = bc.encode(['none','没有','偷东西'])
result2 = bc.encode(['none没有偷东西'])
result3 = bc.encode(['none  没有  偷   东西'])
print(result1)
print(result1.shape)

# result2 与 result3的结果一样，中间用空格分开是一样的
print(result2.shape)
print(result3.shape)

print(result2 == result3)