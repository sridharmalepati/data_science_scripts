import pandas as pd
import numpy as np
import scipy as sp
import re
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

def mungling(data):
    """Data mungling:
        * lower case columns;
        * fill missing values;
        * add new features
    """

    #data = data.drop(['Ticket', 'Embarked'], axis=1)

    data.rename(columns=lambda v: v.lower(), inplace=True)
    data.rename(columns={'passengerid': 'id', 'pclass': 'klass'}, inplace=True)

    # extract title ('Mr.', 'Miss.', etc)
    data['title'] = data.name.map(
            lambda n: re.search('\w+\.', n).group().lower()[:-1])

    # some cabin names start with "F .."
    data.cabin = data.cabin.map(lambda c: str(c).replace('F ', '').lower())

    # extract cabin type (first letter)
    # Nan cabins mark as "n"
    data['cabin_type'] = data.cabin.map(
            lambda c: str(c).split()[0][0])

    data['alone'] = data.sibsp + data.parch
    data.alone = data.alone.map(lambda v: 0 if v > 0 else 1)

    # fill missing age as an mean of a particular group
    data.age = data.groupby(['sex', 'alone', 'title']).age.transform(
            lambda age: age.fillna(age.mean()))

    data.age.fillna(data.age.mean(), inplace=True)

    # for those how both ticket to more then one cabin (traveling with
    # family) price should be divided by numer of cabins
    nums = data.cabin.map(lambda c: len(str(c).split(' ')))
    data.fare = (data.fare / nums)

    # the same for ticket price
    #data.fare = data.groupby(['klass', 'cabin_type', 'age']).fare.transform(
    #   lambda g: g.fare.fillna(g.fare.mean()))
    data.fare.fillna(data.fare.mean(), inplace=True)

    # crew do not pay for the ticket
    data['crew'] = data.fare.map(lambda p: 1 if p == 0 else 0)

    data['sex_num'] = data.sex.replace({'male': 0, 'female': 1})

    return data

def prepare_sets():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Stack train and test data sets together in order
    # to fill missing values using global averages (age, fare).
    # To do so we need to remove Survived from train data frame
    # and mark data in both collections as 'train' an 'test'
    # correspondingly.
    survived = train_data.Survived
    train_data = train_data.drop(['Survived'], axis=1)

    train_data['type'] = 'train'
    test_data['type'] = 'test'

    data = train_data.append(test_data, ignore_index=True)
    data = mungling(data)

    sex = pd.get_dummies(data.sex, prefix='sex')
    cabin = pd.get_dummies(data.cabin_type, prefix='cabin')
    klass = pd.get_dummies(data.klass, prefix='klass')

    #sub_data = data[['age', 'klass', 'alone', 'crew', 'parch', 'sibsp']]
    #sub_data = data[['age', 'klass', 'sex_num', 'fare', 'parch', 'sibsp']]
    sub_data = data[['age', 'klass', 'sex_num', 'fare', 'alone', 'crew']]
    #sub_data = pd.merge(sub_data, sex, left_index=True, right_index=True)
    #sub_data = pd.merge(sub_data, cabin, left_index=True, right_index=True)
    #sub_data = pd.merge(sub_data, klass, left_index=True, right_index=True)

    return data, sub_data, survived

if __name__ == '__main__':

    data, sub_data, survived = prepare_sets()
    test_data = sub_data[data.type == 'test']
    train_data = sub_data[data.type == 'train']

    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    #    train_data, survived, test_size=0.3, random_state=0)

    #params = [{
    #        'n_estimators': [100, 150, 200, 300],
    #        'min_samples_leaf': [1, 3, 5, 10],
    #        'min_samples_split': [2, 5, 8, 11],
    #        'max_depth': [None, 5, 50],
    #        "bootstrap": [True, False],
    #        #'max_features': [None, 'auto', 'sqrt', 'log2']
    #    }]

    params = {
                'n_estimators': [200, 250],
                'learning_rate': [0.1, 0.2, 0.3],
                #'min_samples_leaf': [1, 2],
                'min_samples_split': [2, 3],
                'subsample': [1., 0.9, 0.8, 0.7]
            }
    clf = ensemble.GradientBoostingClassifier(n_estimators=10)
    #clf = ensemble.RandomForestClassifier(n_estimators=10, random_state=1)
    clf = GridSearchCV(clf, params, cv=5, n_jobs=2)
    #clf.fit(X_train, y_train)

    #print clf.best_estimator_

    #y_true, y_pred = y_test, clf.predict(X_test)
    #print classification_report(y_true, y_pred)


    #clf = ensemble.RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
    #            min_samples_split=10, random_state=1, bootstrap=True)
    #clf = ensemble.RandomForestClassifier(n_estimators=200, min_samples_leaf=2, min_samples_split=10, random_state=1)
    clf.fit(train_data, survived)

    print clf.best_estimator_

    predictions = clf.predict(test_data)

    print 'PassengerId,Survived'
    for i in range(test_data.shape[0]):
        print '%s,%s' % (data[data.type=='test'].irow(i)['id'], predictions[i])


    #n_folds = 5
    #predictions = np.zeros((n_folds, test_data.shape[0]))

    #idx = 0

   # for train_idx, test_idx in cross_validation.KFold(
   #         n=train_data.shape[0], n_folds=n_folds):
   #     clf = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
        #clf = ensemble.RandomForestClassifier(n_estimators=300, random_state=1)
   #     clf.fit(train_data.ix[train_idx], survived[train_idx])

   #     predictions[idx] = clf.predict(test_data)
   #     idx += 1

    #tdata = data[data.type == 'test']

    #predictions = predictions.T
    #print 'PassengerId,Survived'
    #for i in range(test_data.shape[0]):
    #    v = 1 if sum(predictions[i]) >= int(round(n_folds / 2.)) else 0
        #print predictions[i], v
    #    row = tdata.irow(i)
    #    print "%s,%s" % (row['id'], v)
