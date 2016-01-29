import sys
import getopt
from images_to_matrices import load_train_data
from images_to_matrices import load_test_data
from images_to_matrices import label_matrices_to_csv
from sklearn.linear_model import LogisticRegression
import numpy as np


def run_logistic_regression(train_size=45000, test=False):
  train_dataset, train_labels = load_train_data()
  train_dataset = train_dataset.reshape((train_dataset.shape[0], train_dataset.shape[1] *
                                         train_dataset.shape[2] * train_dataset.shape[3]))

  # Create a validation dataset
  valid_size = 5000

  valid_dataset = train_dataset[:valid_size, :]
  valid_labels = train_labels[:valid_size]
  train_dataset = train_dataset[valid_size:valid_size + train_size, :]
  train_labels = train_labels[valid_size:valid_size + train_size]
  print 'Training', train_dataset.shape, train_labels.shape
  print 'Validation', valid_dataset.shape, valid_labels.shape

  # Training
  print 'Training...'
  logreg = LogisticRegression()
  logreg.fit(train_dataset, train_labels)

  train_predict = logreg.predict(train_dataset)
  valid_predict = logreg.predict(valid_dataset)

  def accuracy(predictions, labels):
    return 100.0 * np.sum(predictions == labels) / predictions.shape[0]

  print 'Training accuracy: %.2f%%' % accuracy(train_predict, train_labels)
  print 'Validation accuracy: %.2f%%' % accuracy(valid_predict, valid_labels)

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
  run_logistic_regression(train_size=1000, test=True)
