from gensim.models.word2vec import Word2Vec

w2v = Word2Vec.load('w2v_model.pkl')
print(w2v['0'])