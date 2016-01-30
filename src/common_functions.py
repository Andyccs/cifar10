from constants import image_size, color_channel, num_labels
from images_to_matrices import load_train_data
import numpy as np


def get_train_valid_data(train_subset=45000, valid_size=5000):
  if train_subset + valid_size > 50000:
    raise Exception('train_subset + valid_size cannot be more than 50000')
  elif train_subset < 128:
    raise Exception('train_subset cannot be less than 128 (the batch size)')

  train_dataset, train_labels = load_train_data()
  train_dataset = reformat_dataset(train_dataset)
  train_labels = reformat_labels(train_labels)

  # Create a validation dataset
  valid_dataset = train_dataset[:valid_size, :]
  valid_labels = train_labels[:valid_size]
  train_dataset = train_dataset[valid_size:valid_size + train_subset, :]
  train_labels = train_labels[valid_size:valid_size + train_subset]
  print 'Training set size:', train_dataset.shape, train_labels.shape
  print 'Validation set size:', valid_dataset.shape, valid_labels.shape

  return train_dataset, train_labels, valid_dataset, valid_labels


def reformat_dataset(dataset):
  # change the dataset shape from
  # (50000, 32, 32, 3)
  # (50000, 3, 32, 32)
  dataset = np.rollaxis(dataset, 3, 1)
  return dataset.reshape((-1, image_size * image_size * color_channel)).astype(np.float32)


def reformat_labels(labels):
  return (np.arange(num_labels) == labels[:, None]).astype(np.float32)
