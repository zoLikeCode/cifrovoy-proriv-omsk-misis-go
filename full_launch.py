import argparse
from first_task.resume_preprocessor import preprocessing_resume_classifier_test
from second_task.salary_preprocessor import preprocessing_salary_prediction_test

import pandas as pd
from joblib import load

parser = argparse.ArgumentParser(description='Скрипт для предсказания и сохранения результатов модели.')

parser.add_argument('test_file_fisrt', type=str, help='Путь к файлу test.csv для первого задания')

parser.add_argument('test_file_second', type=str, help='Путь к файлу test.csv для второго задания')

parser.add_argument('submission_file_first', type=str, help='Путь для сохранения файла submission.csv первого задания')

parser.add_argument('submission_file_second', type=str, help='Путь для сохранения файла submission.csv второго задания')

args = parser.parse_args()


#Запуск первой задачи
raw_data_resume = pd.read_csv(args.test_file_fisrt)

data_resume = preprocessing_resume_classifier_test(raw_data_resume)

lr = load(f'first_task\\lr_classifier_model_0.joblib')
tfidf = load(f'first_task\\tfidf_model_0.joblib')

submission_resume = pd.DataFrame([])
submission_resume['id'] = data_resume['id']
test_corpus = data_resume['text_prep'].tolist()
X_tf = tfidf.transform(test_corpus).toarray()
y_pred_t = lr.predict(X_tf)
submission_resume['job_title'] = y_pred_t

submission_resume.to_csv(args.submission_file_first + f'\\submissions1.csv', index=False)

#запуск второй задачи
raw_data_sal = pd.read_csv(args.test_file_second)

data_sal = preprocessing_salary_prediction_test(raw_data_sal)

clf = load(f'second_task\\salary_prediction_model_0.joblib')

submission_sal = pd.DataFrame([])
submission_sal['id'] = data_sal['id']
X_test = data_sal.drop('id', axis=1)
y_pred = clf.predict(X_test)
submission_sal['salary'] = y_pred

submission_sal.to_csv(args.submission_file_second + f'\\submissions2.csv', index=False)

def create_submission(RES_part, SAL_part):
  RES_part['task_type'] = 'RES'
  SAL_part['task_type'] = 'SAL'
  submission = pd.concat([RES_part, SAL_part], axis=0)
  submission.to_csv('submission.csv', index=False)

sub1 = pd.read_csv(args.submission_file_first + f'\\submissions1.csv')
sub2 = pd.read_csv(args.submission_file_second + f'\\submissions2.csv')

create_submission(sub1, sub2)