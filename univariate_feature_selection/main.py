
import pandas as pd
import numpy as np
from sklearn import metrics
from data import Data
from joblib import Memory
from sklearn import cross_validation
from sklearn.mixture import GMM
from sklearn.naive_bayes import BernoulliNB

RANDOM_SEED = 10

memory = Memory(cachedir='_cache', verbose=0)

def run_cross_validation(X, y, klass, params, n_folds=5):
    print '\n{folds}-fold cross validation ({clf}, {params})'.format(
                                    folds=n_folds, clf=klass, params=params)
    print 'X shape:', X.shape

    kf = cross_validation.KFold(X.shape[0], n_folds=n_folds, shuffle=True,
                                random_state=RANDOM_SEED)

    clf = klass(**params)

    scores = []
    for idx, (train, test) in enumerate(kf):
        X_train, X_test = X[train, :], X[test, :]
        y_train, y_test = y[train], y[test]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        scores.append(metrics.f1_score(y_test, pred))
        print '[%s] f1 score: %s' %(idx + 1, scores[-1])

    print 'mean f1 score:', np.mean(scores)

def main():
    data = Data()

    train_bimodal = data.get_train(False, True, False)
    train_bin = data.get_train(True, False, False)

    params = {'n_components': 2, 'random_state': RANDOM_SEED}
    run_cross_validation(train_bimodal.values, data.labels, GMM, params)

    run_cross_validation(train_bin.values, data.labels, BernoulliNB, {})

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    main()
