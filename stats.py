from sklearn import linear_model
import numpy as np
import pandas as pd
import random
import abcd




FEATURES = ['num_errors', 'essay_length', 'num_words', 'avg_word_len', 'num_punc']

def linearRegression(x, y):
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    # clf.predict([])
    # print(clf.coef_, clf.intercept_)
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


def test():
    metatrain = pd.read_csv('Data/train_features.tsv', sep='\t')
    metatest = pd.read_csv('Data/test_features.tsv', sep='\t')
    res = pd.DataFrame(columns=['essay_id', 'essay_set', 'prediction'])

    for setID in range(1, 9):
        data = metatrain.loc[metatrain['essay_set'] == setID]
        test = metatest.loc[metatrain['essay_set'] == setID]
        test_essayID = test['essay_id']
        test_essaySet = test['essay_set']
        score = data['actual']

        score_min = data['actual'].min()
        score_max = data['actual'].max()
        print("min: ",score_min," max:",score_max)

        train = normalization(data)
        test = normalization(test)
        train_y = train['actual']

        train = train.loc[:, FEATURES]
        test = test.loc[:, FEATURES]
        # normal_train = normalization(train)
        # normal_test = normalization(test)

        train = train.values

        clf = linearRegression(train, train_y)

        predicted = clf.predict(test.values)

        predicted = np.around(predicted * (score_max - score_min) + score_min)
        print(predicted.astype(int))
        # print(score_test.values)
        test['prediction'] = predicted.astype(int)
        test['essay_id'] = test_essayID
        test['essay_set'] = test_essaySet

        print("---------------------------------------------")
        print(test.loc[:, ['essay_id', 'essay_set', 'prediction']])
        res = pd.concat([res, test.loc[:, ['essay_id', 'essay_set', 'prediction']]])


    res.to_csv('Data/result.tsv', sep='\t', index=False)
        # evaluate = abcd.QWK(predicted.astype(int), score_test.values, score_max)

def train():
    metatrain = pd.read_csv('Data/train_features.tsv', sep='\t')

    setID = 1

    data = metatrain.loc[metatrain['essay_set'] == setID]
    score = data['actual']
    score_min = data['actual'].min()
    score_max = data['actual'].max()

    # print(score_min)

    data = normalization(data)

    index = np.random.choice([0,1], size=len(data), replace=True, p=[0.2, 0.8])
    # print(index)
    # print(np.logical_not(index))

    train = data.loc[index==1]
    score_train = score.loc[index==1]
    score_test = score.loc[index==0]

    train_y = train['actual']
    test = data.loc[index==0]
    test_y = test['actual']


    print('---------------------train------------------')
    print(train)
    print('---------------------test------------------')
    print(test)


    # features = train.loc[:, ['num_errors', 'essay_length', 'num_words']]
    train = train.loc[:, FEATURES]
    test = test.loc[:, FEATURES]
    # normal_train = normalization(train)
    # normal_test = normalization(test)

    train = train.values


    clf = linearRegression(train, train_y)

    predicted = clf.predict(test.values)

    predicted = np.around(predicted * (score_max-score_min) + score_min)
    print(predicted.astype(int))
    print(score_test.values)
    evaluate = abcd.QWK(predicted.astype(int), score_test.values,score_max)

if __name__ == '__main__':
    test()