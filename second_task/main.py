import argparse

parser = argparse.ArgumentParser(description='Скрипт для предсказания и сохранения результатов модели.')

parser.add_argument('test_file', type=str, help='Путь к файлу test.csv')

parser.add_argument('submission_file', type=str, help='Путь для сохранения файла submission.csv')

args = parser.parse_args()
