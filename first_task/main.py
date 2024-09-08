import pandas as pd
import argparse
from joblib import load
from resume_preprocessor import preprocessing_resume_classifier_test

parser = argparse.ArgumentParser(description='Скрипт для предсказания и сохранения результатов модели.')

parser.add_argument('test_file', type=str, help='Путь к файлу test.csv')

parser.add_argument('submission_file', type=str, help='Путь для сохранения файла submission.csv')

args = parser.parse_args()

raw_data = pd.read_csv(args.test_file)

data = preprocessing_resume_classifier_test(raw_data)

lr = load('lr_classifier_model_0.joblib')
tfidf = load('tfidf_model_0.joblib')

submission = pd.DataFrame([])
submission['id'] = data['id']
test_corpus = data['text_prep'].tolist()
X_tf = tfidf.transform(test_corpus).toarray()
y_pred_t = lr.predict(X_tf)
submission['job_title'] = y_pred_t

submission.to_csv(args.submission_file + f'\\submissions1.csv', index=False)

