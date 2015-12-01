mport pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from joblib import Memory

memory = Memory(cachedir='_cache', verbose=0)

@memory.cache
def get_data():
    # since tsv stands for "tab separated values" lets use \t as a delimiter
    train = pd.read_csv('../data/train.tsv', delimiter='\t')
    test = pd.read_csv('../data/test.tsv', delimiter='\t')
    return train, test

class Data(object):

    def __init__(self):
        train, test = get_data()

        self.labels = train.y
        self.train_id = train.id
        self.test_id = test.id
        self.test = test.drop('id', axis=1)
        self.train = train.drop(['id', 'y'], axis=1)

        self.__group_columns(self.train)

        self.columns = self.train.columns.tolist()

    def __group_columns(self, data):
        """Split columns in two groups:
           - float values
           - 0/1 values
           used later by filter_columns
        """

        dtypes = data.dtypes
        # except cId
        self.int_columns = dtypes[dtypes == np.int64].index[1: ].tolist()
        self.float_columns = dtypes[dtypes == np.float64].index.tolist()

    def get_train(self, int_cols=True, float_cols=True, cid=True):
        return self.filter_columns(self.train, int_cols, float_cols, cid)

    def get_test(self, int_cols=True, float_cols=True, cid=True):
        return self.filter_columns(self.test, int_cols, float_cols, cid)

    def train_test_split(self, int_cols=True, float_cols=True, cid=True):
        assert int_cols or float_cols or cid

        x_train, x_test, y_train, y_test = train_test_split(self.train,
                                                            self.labels,
                                                            test_size=0.25,
                                                            random_state=10)

        x_train = self.filter_columns(x_train, int_cols, float_cols, cid)
        x_test = self.filter_columns(x_test, int_cols, float_cols, cid)

        return x_train, x_test, y_train, y_test

    def filter_columns(self, data, int_cols=True, float_cols=True, cid=True):
        cols = []
        if cid:
            cols.append('cId')

        if int_cols:
            cols.extend(self.int_columns)

        if float_cols:
            cols.extend(self.float_columns)

        return data[cols]

if __name__ == '__main__':

    data = Data()

    x_train, x_test, y_train, y_test = data.train_test_split(True, False, True)
    x_train2, _, _, _ = data.train_test_split(True, False, True)

    all_cols = set(x_train.columns.tolist()).symmetric_difference(
                   x_train2.columns.tolist())

    assert not all_cols

    train = data.get_train(False, True, False)
    print train.columns
