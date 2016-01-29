from common_functions import reformat_dataset, reformat_labels
from images_to_matrices import label_matrices_to_csv
from images_to_matrices import load_test_data
from images_to_matrices import load_train_data
import numpy as np


def run_multinomial_logistic_regression():
  train_dataset, train_labels = load_train_data()
  train_dataset = reformat_dataset(train_dataset)
  train_labels = reformat_labels(train_labels)


if __name__ == '__main__':
  run_multinomial_logistic_regression()
