import pandas as pd
from grammarbot import GrammarBotClient
import time

def test_grammarbot():
    client = GrammarBotClient(api_key='KS9C5N3Y')
    text = 'I are beautiful. There are two apple. I are smart. what a woderful day!'
    res = client.check(text)

    print(len(res.matches))
    for i in range(len(res.matches)):
        print(res.matches[i].corrections)


"""
    Using GrammarBot to extract grammar features.API key is KS9C5N3Y.
"""
def grammar_check(text):
    client = GrammarBotClient(api_key='KS9C5N3Y')
    return len(client.check(text).matches)


"""
    features: error length words
"""
def feature_extract(df):
    features = []
    for index, row in df.iterrows():
        text = row['essay']
        try:
            error = grammar_check(text)
        except:
            error = -1
            print('grammar check failed')
        length = len(text)
        words = len(text.split())
        essay_id = row['essay_id']
        essay_set = row['essay_set']
        actual = row['domain1_score']

        features.append([essay_id, essay_set, error, length, words, actual])
    return pd.DataFrame(features, columns=['essay_id', 'essay_set', 'num_errors', 'essay_length', 'num_words', 'actual'])


if __name__ == "__main__":
    df = pd.read_csv("Data/train.tsv", sep='\t', usecols=['essay_id', 'essay_set', 'essay', 'domain1_score'])
    df_features = feature_extract(df)
    df_features.to_csv("Data/features.tsv", sep='\t', columns=['essay_id', 'essay_set', 'num_errors',
                                                                     'essay_length', 'num_words', 'actual'])
