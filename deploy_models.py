# встроенные модули
import string
import re
import pickle

# необходимые зависимости, но желательно еще иметь установленный sklearn
from nltk.corpus import stopwords
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    PER,
    NamesExtractor,
    MorphVocab,
    Doc,
    NewsNERTagger
)


# класс предобработки текстовых данных
class PreprocessData():
    def __init__(self):

        # чистка списка стоп-слов
        self.russian_stopwords = stopwords.words('russian')
        words_to_del = ['хорошо']
        words_to_add = ['rating', 'ооо', 'оао', 'ао', "ра", 'пао', 'гк', 'ппк', 'зао', 'нкр', "акр", "акра", "аналитик", 'компания', 'эксперт','рейтинг', 'агенство', 'млрд', 'руб', 'м', 'х', 'нпк', 'овк', 'г', 'гг', 'i', 'ii']
        self.russian_stopwords.extend(words_to_add)

        for w in words_to_del:
            self.russian_stopwords.remove(w)

        # эмбеддинги для Наташи
        emb = NewsEmbedding()

        # сегментеры и таг-выделители сущностей/морфем для анализа слов
        self.segmenter = Segmenter()
        self.morph_tagger = NewsMorphTagger(emb)
        self.morph_vocab = MorphVocab()

        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.ner_tagger = NewsNERTagger(emb)

    # удаление всех компаний
    def delete_company(self, s):
        pattern = r'«.*?»'
        return re.sub(pattern, '', s)

    # удаление всех сайтов
    def delete_sites(self, s):
        sites = r"(?:https?:\/\/|ftps?:\/\/|www\.)\S"
        emails='([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'
        r =  re.sub(sites, '', s)
        return re.sub(emails, '~', r)

    # удаление всей пунктуации
    def delete_punkt(self, s):
        return s.translate(str.maketrans('', '', string.punctuation + "«»—№–"))

    # удаление всех чисел
    def delete_digits(self, s):
        return re.sub("[0-9]", "", s)

    # удаление всех стоп-слов
    def delete_stopwords(self, tokens, names):
        without_stopwords = [t for t in tokens
                             if t not in self.russian_stopwords and t not in names]
        return " ".join(without_stopwords)

    # нормализовация текстов
    def normalized(self, text):
        doc = Doc(text)

        # сегментация и морфемный анализ
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        lemmatized_sentence_list = []

        # приведение к начальным формам
        doc.tag_ner(self.ner_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemmatized_sentence_list.append(token.lemma)

        # избавление от именновых сущностей (имена личностей)
        names = []
        for span in doc.spans:
            if span.type == PER:
                span.normalize(self.morph_vocab)
                names.extend(span.normal.lower().split())

        # удаление стоп-слов вместе с именами сразу
        text = self.delete_stopwords(lemmatized_sentence_list, names)
        return text

    # предобработка
    def preprocess(self, text):
        text_without_company = self.delete_company(text)
        text_without_sites = self.delete_sites(text_without_company)
        text_without_punct = self.delete_punkt(text_without_sites)
        text_without_digits = self.delete_digits(text_without_punct)
        return self.normalized(text_without_digits)

# распаковка модели векторизации и модели регресси
with open(VEC_MODEL_PATH, 'rb') as vec_model, open(MODEL_PATH, 'rb') as model:
    vec_model = pickle.load(vec_model)
    model = pickle.load(model)

# объявляем только один раз класс предобработки
PreData = PreprocessData()

# обработка строки пользователя
preprocessed_user_string = PreData.preprocess(USERSTRING)

# векторизация строки
vector = vec_model.transform(preprocessed_user_string.)
# предсказание на строке (одно число float)
prediction = model.predict([vector])

# словарь диапазонов значений числовой шкалы и шкалы уровней рейтинга
values_dict = {0.0: 'C',
               0.04999999999999999: 'C',
               0.09999999999999998: 'C',
               0.14999999999999997: 'B-',
               0.19999999999999996: 'B',
               0.25: 'B+',
               0.3: 'BB-',
               0.35: 'BB',
               0.4: 'BB+',
               0.45: 'BBB-',
               0.5: 'BBB',
               0.55: 'BBB+',
               0.6: 'A-',
               0.65: 'A',
               0.7: 'A+',
               0.75: 'AA-',
               0.8: 'AA',
               0.8500000000000001: 'AA+',
               0.9: 'AAA',
               0.95: 'AAA',
               1.0: 'AAA'}

# ф-ция для восстановления уровня рейтинга по предсказанию
def return_rating_levels(value, error, values_dict):
    for k, v in values_dict.items():
        if abs(value - k) < error:
            return v

# ф-ция для восстановления категории по предсказанию
def return_category_classes(value):
    return value.strip('+-')

# ф-ция вычисления "уверенности" в предсказании
def compute_certanity(dist):
    return np.clip(math.cos(dist/31.9*1000), 0, 1)

# значение "уверенности" в предсказании
def get_class(prediction, values_dict):
    """
    :param prediction: результат работы модели
    :param values_dict: словарь весов
    :return: возвращает два кортежа, каждый с детальным рейтингом и уверенностью в процентах
    если второй и первый  по уверенности рейтинги это AAA или C, возвращается только один кортеж
    """
    # список ключей, отсортированных по их абсолютному отличию от n
    sorted_keys = sorted(values_dict.keys(), key=lambda x: abs(x - prediction))

    # ближайший ключ - это первый элемент в отсортированном списке
    closest_key = sorted_keys[0]

    # второй ближайший ключ - это второй элемент в отсортированном списке
    second_closest_key = sorted_keys[1]
    if values_dict[closest_key]==values_dict[second_closest_key]:
        return (values_dict[closest_key], round(compute_certanity(closest_key-prediction)*100, 2))
    
    return (values_dict[closest_key], round(compute_certanity(closest_key-prediction)*100, 2)), 
           (values_dict[second_closest_key], round(compute_certanity(second_closest_key-prediction)*100, 2))

# диапазон отклонения/погрешности для восстановления уровня рейтинга
error = 0.025

# итоговый кортеж из четырех элементов
# 1: предсказанная моделью категория [string]
# 2: предсказанный моделью уровень рейтинга [string]
# 3: кортеж с одним/двумя ближайшими уровнями рейтинга вида tuple(уровень рейтинга[string], степень близости[float])
# 4: изначально полученное предсказанное значение [float] (вдруг понадобится)
prediction_tuple = (return_category_classes(prediction), 
                    return_rating_levels(prediction, error, values_dict),
                    get_class(prediction, values_dict), 
                    prediction)
