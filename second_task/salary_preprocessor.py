import pandas as pd

from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
import string
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing_salary_prediction_test(raw_data):
  #отбор наиболее релевантных колонок
  features_name = ['id', 'required_experience', 'vacancy_address_latitude', 'vacancy_address_longitude', 'professionalSphereName', 'position_requirements']

  raw_data = raw_data[features_name]
  vacancies_copy = raw_data.copy()

  # NLP препроцессинг поля position_requirements
  def remove_punctuation(text):
    return ''.join([ch if ch not in string.punctuation else ' ' for ch in text])
  def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])
  def remove_eng(text):
    return ''.join([' ' if i[0] in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] else i for i in text])
  def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)

  prep_text = [remove_multiple_spaces(remove_eng(remove_numbers(remove_punctuation(text.lower())))) for text in vacancies_copy['position_requirements'].astype('str')]

  
  # Кодируем position_requirements c помощью преобразования TF-IDF
  russian_stopwords = stop_words
  tfidf = TfidfVectorizer(max_features=5, stop_words=russian_stopwords)
  skills_encoded = tfidf.fit_transform(prep_text)
  skills_encoded = pd.DataFrame(skills_encoded.toarray(), columns=tfidf.get_feature_names_out()).iloc[:, 1:]

  vacancies_copy = pd.concat([vacancies_copy, skills_encoded], axis=1)
  vacancies_copy.drop('position_requirements', axis=1, inplace=True)
  
  # Стандартизируем числовые признаки
  scaler = StandardScaler()
  vacancies_copy[['required_experience', 'vacancy_address_latitude', 'vacancy_address_longitude']] = scaler.fit_transform(vacancies_copy[['required_experience', 'vacancy_address_latitude', 'vacancy_address_longitude']])

  final = vacancies_copy.copy()
  # One-hot encoding поля professionalSphereName
  final = pd.get_dummies(final, columns=['professionalSphereName'])
  final = final.drop(['professionalSphereName_Маркетинг, реклама, PR'], axis=1)

  return final