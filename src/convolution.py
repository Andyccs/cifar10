from common_functions import get_train_valid_data


def run_convolution(train_subset=45000, valid_size=5000, test=False):
  train_dataset, train_labels, valid_dataset, valid_labels = \
      get_train_valid_data(train_subset, valid_size)

  print 'Building graph...'


if __name__ == '__main__':
  run_convolution(train_subset=45000, valid_size=5000, test=False)
