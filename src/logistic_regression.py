from common_functions import reformat_dataset
from images_to_matrices import label_matrices_to_csv
from images_to_matrices import load_test_data
from images_to_matrices import load_train_data
from sklearn.linear_model import LogisticRegression
import getopt
import numpy as np
import sys


def accuracy(predictions, labels):
  return 100.0 * np.sum(predictions == labels) / predictions.shape[0]


def run_logistic_regression(train_subset=45000, valid_size=5000, test=False):
  train_dataset, train_labels = load_train_data()
  train_dataset = reformat_dataset(train_dataset)

  valid_dataset = train_dataset[:valid_size, :]
  valid_labels = train_labels[:valid_size]
  train_dataset = train_dataset[valid_size:valid_size + train_subset, :]
  train_labels = train_labels[valid_size:valid_size + train_subset]
  print 'Training set size: ', train_dataset.shape, train_labels.shape
  print 'Validation set size: ', valid_dataset.shape, valid_labels.shape

  print 'Training...'
  logreg = LogisticRegression()
  logreg.fit(train_dataset, train_labels)

  train_predict = logreg.predict(train_dataset)
  valid_predict = logreg.predict(valid_dataset)

  train_accuracy = accuracy(train_predict, train_labels)
  valid_accuracy = accuracy(valid_predict, valid_labels)
  print_accuracy(train_accuracy, valid_accuracy)

  # Predict test data
  if (not test):
    return

  print 'Predicting test dataset...'
  test_dataset = load_test_data()
  test_dataset = test_dataset.reshape((test_dataset.shape[0], test_dataset.shape[1] *
                                       test_dataset.shape[2] * test_dataset.shape[3]))

  test_predict = logreg.predict(test_dataset)
  label_matrices_to_csv(test_predict, 'submission.csv')


if __name__ == '__main__':
  run_logistic_regression(train_subset=1000, test=False)
