import pandas as pd
from grammarbot import GrammarBotClient
import time
import language_check
import proselint

PUNCTUATION = [',', '.', '?', '!', '\'', '\"', '(', ':', ';']
COLUMNS = ['essay_id', 'essay_set', 'num_errors', 'essay_length', 'num_words','avg_word_len', 'num_punc', 'actual']
from spellchecker import SpellChecker

"""
    Example: def grammercheck("This is a apple") return number of grammar errors : 1
    text is a string, which contains a essay.
"""
def add_any_new_feature(text):
    pass



def num_punctuation(text):
    res = 0
    for i in text:
        if i in PUNCTUATION:
           res += 1
    return res

def avgWordlen(text):
    arr = str.split(text)
    length = 0
    for word in arr:
        length += len(word)
    return length / len(arr)

def spell_checker(text):
    arr = str.split(text)
    for index, word in enumerate(arr):
        if word[-1] in ['.',',','?','!',':']:
            arr[index] = word[:-1]
    spell = SpellChecker()
    misspelled = spell.unknown(arr)
    # print(misspelled)

    return len(misspelled)

# def test_grammarbot():
#     client = GrammarBotClient(api_key='KS9C5N3Y')
#     text = 'I are beautiful. There are two apple. I are smart. what a woderful day!'
#     res = client.check(text)
#
#     print(len(res.matches))
#     for i in range(len(res.matches)):
#         print(res.matches[i].corrections)
#
#


"""
    features: error length words
"""
def feature_extract(df):
    features = []

    for index, row in df.iterrows():

        text = row['essay']
        try:
            error = spell_checker(text)
        except:
            error = -1
            print('spell check failed')
        length = len(text)
        words = len(text.split())
        avg_word_len = avgWordlen(text)
        num_punc = num_punctuation(text)
        essay_id = row['essay_id']
        essay_set = row['essay_set']
        try:
            actual = row['domain1_score']
        except:
            actual = 0

        features.append([essay_id, essay_set, error, length, words,avg_word_len, num_punc, actual])

    return pd.DataFrame(features, columns=COLUMNS)


if __name__ == "__main__":
    df = pd.read_csv("Data/test.tsv", sep='\t', usecols=['essay_id', 'essay_set', 'essay'])
    df_features = feature_extract(df)
    df_features.to_csv("Data/test_features.tsv", sep='\t', columns=COLUMNS)

