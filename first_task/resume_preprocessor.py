import string
import re

def preprocessing_resume_classifier_test(raw_data):
  # отбор полезных колонок
  raw_data = raw_data[['id', 'demands']]

  # NLP препроцессинг резюме
  def remove_punctuation(text):
    return ''.join([ch if ch not in string.punctuation else ' ' for ch in text])
  def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])
  def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)

  prep_text = [remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))) for text in raw_data['demands'].astype('str')]
  raw_data['text_prep'] = prep_text

  final = raw_data.copy()
  return final