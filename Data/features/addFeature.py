from os import listdir
import pandas as pd
import stats

import os
files = listdir(os.getcwd())
print(files)
res = pd.DataFrame(columns=['essay_id', 'essay_set', 'prediction'])

for setID in range(1, 9):
    curr_res = pd.DataFrame(columns=['essay_id', 'essay_set', 'prediction'])
    train_file = "features_set_train_" + str(setID) + ".csv"
    test_file = "features_set_test_" + str(setID) + ".csv"

    train_pd = pd.read_csv(train_file, sep=',')
    train_pd = train_pd.dropna(axis=1)
    train_data = train_pd.iloc[:, 4:-1]
    # data = metatrain.loc[metatrain['essay_set'] == id]
    train_meta = pd.read_csv("../train.csv", sep=',', usecols=['domain1_score'])
    train_score = train_meta['domain1_score']
    
    # test_meta = pd.read_csv("../test.csv", sep=',', usecols=['domain1_score'])
    # test_score = test_meta['domain1_score']

    test_pd = pd.read_csv(test_file, sep=',')
    test_pd = test_pd.dropna(axis=1)
    test_data = test_pd.iloc[:, 4:-1]
    
    train_2gram = pd.read_csv("../tf_idf_2gram.csv", sep='\t')
    train_data['2gram'] = train_2gram['2gram-tf-idf']
    
    test_2gram = pd.read_csv("../tf_idf_2gram.csv", sep='\t')
    test_data['2gram'] = test_2gram['2gram-tf-idf']

    train_3gram = pd.read_csv("../tf_idf_3gram.csv", sep='\t')
    train_data['3gram'] = train_3gram['3gram-tf-idf']

    test_3gram = pd.read_csv("../tf_idf_3gram.csv", sep='\t')
    test_data['3gram'] = test_3gram['3gram-tf-idf']

    clf = stats.knnRegression()
    clf.fit(train_data, train_score)
    predicted = clf.predict(test_data)



    curr_res['prediction'] = predicted.astype(int)
    curr_res['essay_id'] = test_pd['essay_id']
    curr_res['essay_set'] = test_pd['essay_set']


    res = pd.concat([res, curr_res.loc[:, ['essay_id', 'essay_set', 'prediction']]])
   

res.to_csv('../features_set_test_all.csv', sep=',', index=False)