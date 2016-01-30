from constants import image_size, color_channel, num_labels
from images_to_matrices import load_train_data
import matplotlib.pyplot as plt
import numpy as np


def print_loss(train_loss, validation_loss):
  print 'Training loss:', train_loss
  print 'Valid loss at step:', validation_loss


def print_accuracy(train_accuracy, validation_accuracy):
  print 'Training accuracy: %.2f%%' % train_accuracy
  print 'Validation accuracy: %.2f%%' % validation_accuracy


def plot(v1, v2, xlabel, ylabel):
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.plot(range(len(v1)), v1, color='g', label='v1')
  plt.plot(range(len(v2)), v2, color='r', label='v2')
  plt.show()


def accuracy(predictions, labels):
  """
  Calculate the accuracy of two (n X num_labels) matrices
  """
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def get_train_valid_data(train_subset=45000, valid_size=5000, reformat_data=True, reformat_label=True):
  if train_subset + valid_size > 50000:
    raise Exception('train_subset + valid_size cannot be more than 50000')
  elif train_subset < 128:
    raise Exception('train_subset cannot be less than 128 (the batch size)')

  train_dataset, train_labels = load_train_data()

  if reformat_data:
    train_dataset = reformat_dataset(train_dataset)

  if reformat_label:
    train_labels = reformat_labels(train_labels)

  train_dataset = train_dataset.astype(np.float32)
  train_labels = train_labels.astype(np.float32)

  # Create a validation dataset
  valid_dataset = train_dataset[:valid_size]
  valid_labels = train_labels[:valid_size]
  train_dataset = train_dataset[valid_size:valid_size + train_subset]
  train_labels = train_labels[valid_size:valid_size + train_subset]
  print 'Training set size:', train_dataset.shape, train_labels.shape
  print 'Validation set size:', valid_dataset.shape, valid_labels.shape

  return train_dataset, train_labels, valid_dataset, valid_labels


def reformat_dataset(dataset):
  # change the dataset shape from
  # (50000, 32, 32, 3)
  # (50000, 3, 32, 32)
  dataset = np.rollaxis(dataset, 3, 1)
  return dataset.reshape((-1, image_size * image_size * color_channel))


def reformat_labels(labels):
  return (np.arange(num_labels) == labels[:, None])
