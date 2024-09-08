from salary_preprocessor import preprocessing_salary_prediction_test
import pandas as pd
import argparse
from joblib import load

parser = argparse.ArgumentParser(description='Скрипт для предсказания и сохранения результатов модели.')

parser.add_argument('test_file', type=str, help='Путь к файлу test.csv')

parser.add_argument('submission_file', type=str, help='Путь для сохранения файла submission.csv')

args = parser.parse_args()

raw_data = pd.read_csv(args.test_file)

data = preprocessing_salary_prediction_test(raw_data)

clf = load('salary_prediction_model_0.joblib')

submission = pd.DataFrame([])
submission['id'] = data['id']
X_test = data.drop('id', axis=1)
y_pred = clf.predict(X_test)
submission['salary'] = y_pred

submission.to_csv(args.submission_file + f'\\submissions2.csv', index=False)

