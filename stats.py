from sklearn import linear_model
import numpy as np
import pandas as pd
import random
import abcd
def linearRegression(x, y):
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    # clf.predict([])
    print(clf.coef_, clf.intercept_)
    return clf


"""
    input:  dataframe
    output: normalized dataframe
"""

def normalization(data):
    return (data - data.min()) / (data.max() - data.min() + 1e-5)

def cross_validation(data):
    # source = data.sample(n=int(len(data)*0.9), random_state=1)
    index = np.random.choice(a=[0, 1], size=len(data), replace=True, p=[0.1, 0.9])
    print(data.loc[index])
    train = data.loc[index]
    test = data.loc[~index]

if __name__ == '__main__':
    data = pd.read_csv('Data/small_features.tsv', sep='\t')
    score_min = data['actual'].min()
    score_max = data['actual'].max()

    print(score_min)
    score =
    data = normalization(data)

    index = np.random.choice([0,1], size=len(data), replace=True, p=[0.2, 0.8])
    print(index)
    print(np.logical_not(index))

    train = data.loc[index==1]
    train_y = train['actual']
    test = data.loc[np.logical_not(index)==1]
    test_y = test['actual']

    print('---------------------train------------------')
    print(train)
    print('---------------------test------------------')
    print(test)


    # features = train.loc[:, ['num_errors', 'essay_length', 'num_words']]
    train = train.loc[:, ['num_errors', 'essay_length']]
    test = test.loc[:, ['num_errors', 'essay_length']]
    # normal_train = normalization(train)
    # normal_test = normalization(test)

    train = train.values


    clf = linearRegression(train, train_y)
    print(test)
    predicted = clf.predict(test.values)

    print(predicted * (score_max-score_min) + score_min)

    # evaluate = abcd.QWK(predicted, actual)