from constants import image_size, color_channel, num_labels
import numpy as np


def reformat_dataset(dataset):
  # change the dataset shape from
  # (50000, 32, 32, 3)
  # (50000, 3, 32, 32)
  dataset = np.rollaxis(dataset, 3, 1)
  return dataset.reshape((-1, image_size * image_size * color_channel)).astype(np.float32)


def reformat_labels(labels):
  return (np.arange(num_labels) == labels[:, None]).astype(np.float32)
