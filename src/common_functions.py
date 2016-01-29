from constants import image_size, color_channel, num_labels
import numpy as np


def reformat_dataset(dataset):
  return dataset.reshape((-1, image_size * image_size * color_channel)).astype(np.float32)


def reformat_labels(labels):
  return (np.arange(num_labels) == labels[:, None]).astype(np.float32)
