import numpy as np

class BatchGenerator:
    def __init__(self,X,y,shuffle=False):
        if not isinstance(X,np.ndarray):
            X = np.asarray(X)   # 当原X发生改变，X的输出也会跟着变
        if not isinstance(y,np.ndarray):
            y = np.asarray(y)

        self._x = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._x.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._x = self._x[new_index]
            self._y = self._y[new_index]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            self._epochs_completed += 1
            if self._shuffle :
                new_index = np.random.permutation(self._number_examples)
                self._x = self._x[new_index]
                self._y = self._y[new_index]
                start = 0
                self._index_in_epoch =batch_size
                assert batch_size <= self._number_examples
            end = self._index_in_epoch
            return self._x[start:end], self._y[start:end]