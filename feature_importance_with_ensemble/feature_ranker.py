
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

def feature_importance(train, labels, n_estimators=50, random_state=10):
    clf = ExtraTreesClassifier(random_state=random_state,
                               n_estimators=n_estimators)
    clf.fit(train, labels)

    importances = clf.feature_importances_

    indices = np.argsort(importances)[::-1]

    print 'Feature ranking:'

    for f in range(train.shape[1]):
        print '%d. feature %s (%f)' % (f + 1, train.columns[indices[f]],
                                       importances[indices[f]])
