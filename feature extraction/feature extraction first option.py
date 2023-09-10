import pickle
import pandas as pd
import numpy as np
import re
import string

from nltk.stem.snowball import SnowballStemmer
import pymorphy2

stemmer = SnowballStemmer("russian")
morph = pymorphy2.MorphAnalyzer()


def get_features(vector, model, vec_model, k):
    vector = vector.toarray()[0]
    names = vec_model.get_feature_names_out()
    coefs = model.coef_

    not_null_values = [(i, v, coefs[i]) for i, v in enumerate(vector) if v != 0]
    top_100 = sorted(not_null_values, key=lambda x: abs(x[2]), reverse=True)[:100]
    words = [names[ind] for ind, tf_idf, c in top_100]
    adj = [w for w in words if re.search(r'\w+[ыио]й(?!\w)', w)]

    if len(adj) >= k * (2 / 3):
        res = adj[:round(k * (2 / 3))] + [w for w in words if w not in adj][:round(k / 3)]
    else:
        res = adj + [w for w in words if w not in adj][:k - len(adj)]

    return res

# нормализация одного слова
def normalized(word):
    word_no_punct = word.translate(str.maketrans('', '', string.punctuation + "«»—№–"))
    word_lem = morph.parse(word.lower())[0].normal_form
    word_stem = stemmer.stem(word_lem)
    return stemmer.stem(word_no_punct.lower())


# возвращение сырых слов из основго предложения
def return_important_substrings(sentence, features):
    result = []
    for feature in features:
        flag = False
        f = [stemmer.stem(_) for _ in feature.split()]
        sequence = []

        for token in sentence.split():
            t = normalized(token)

            # True - начать собирать последовательность
            if t.startswith(f[0]) or f[0] in t:

                if flag:
                    flag = False
                    sequence.clear()
                else:
                    flag = True

            if flag:
                sequence.append(token)

                if f[-1] in t or len(sequence) > 6:
                    flag = False
                    break

        seq = ' '.join(sequence)
        if seq:
            result.append(seq)

    return result

features = get_features(VECTOR, model, vec_model, 40)
result = return_important_substrings(YOURSTRING, features)