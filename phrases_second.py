import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import sentence_transformers

model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
model_cossim = sentence_transformers.SentenceTransformer('inkoziev/sbert_synonymy')


def get_sentiment(text, model, token_len=5, step=5, CRS='AA', sort_by=['positive', 'neutral']):
    df_score = pd.DataFrame(columns=['batch', 'positive', 'neutral', 'negative'])

    # get all tokens with (token_len)-batch size and step window-size
    splitted = text.split()
    tokens = [' '.join(splitted[i:i + token_len]) for i in range(0, len(splitted), step)]

    # extract windows where positive_score > n or neutral_score > k while positive_score > negative_score
    tkn_dict = {}
    for token in tokens:
        sentiment_dict = model(token, top_k=3)
        answ = {}
        for i, d in enumerate(sentiment_dict):
            answ[d['label']] = d['score']

            # in different cases we look for either positive or negative sentiment
        if CRS[0] == 'A':
            pos_score = 0.6; neut_score = 0.85; neg_score = 0.2;
            if (answ['neutral'] >= neut_score and (answ['negative'] <= 2 * answ['positive'])) \
                    or answ['positive'] >= pos_score:
                tkn_dict[token] = answ
                sort_by = ['positive', 'neutral']
        elif len(CRS) > 1 and CRS[0] == 'B':
            neut_score = 0.85
            if answ['neutral'] >= neut_score:
                tkn_dict[token] = answ
        else:
            pos_score = 0.2; neut_score = 0.85; neg_score = 0.6;
            if (answ['neutral'] >= neut_score and (2 * answ['negative'] >= answ['positive'])) \
                    or answ['negative'] >= neg_score:
                tkn_dict[token] = answ
                sort_by = ['negative', 'neutral']

        df_score.loc[len(df_score)] = [token, answ['positive'], answ['neutral'], answ['negative']]
        df_score = df_score.sort_values(by=sort_by, ascending=False)

    return tkn_dict, df_score


# top_k - top k of tokens sorted by sentiment
# top_n - top n matched phrases sorted by cos_sim
def locate(raw_text, tokens, top_k=20, top_n=20):
    df_cossim = pd.DataFrame(columns=['cossim', 'token', 'raw_text'])
    top_tokens = tokens[1:top_k + 1]
    sentences = raw_text.split('.')
    embeddings = model_cossim.encode(sentences)

    raw_text_located = []
    for token in top_tokens:
        v1 = model_cossim.encode(token)
        max_cossim = 0
        max_match = ''
        for i2 in range(len(embeddings)):
            s = sentence_transformers.util.cos_sim(a=v1, b=embeddings[i2]).item()
            if s > max_cossim:
                max_cossim = s;
                max_match = sentences[i2]
        raw_text_located.append({f'{token}': max_match})

        df_cossim.loc[len(df_cossim)] = [s, token, max_match]

    df_cossim = df_cossim.sort_values(by=['cossim'], ascending=False)[1:top_n + 1]
    return raw_text_located, df_cossim

# в CRS аргумент надо передать укрупненную категорию, которую предсказала моделька
# PREPROCESSED_TEXT - обработанный текст из deploy_model, 149я строка деплоя preprocessed_user_string
# RAW_TEXT - текст пресс-релиза
tkn_dict, df_score = get_sentiment(PREPROCESSED_TEXT, model, CRS=crs, token_len=7, step=7)
located_dict, located_df = locate(RAW_TEXT, df_score.batch.to_list(), top_n=7)
key_phrases_raw = located_df.raw_text.to_list()