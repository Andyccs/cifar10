from common_functions import get_train_valid_data


def run_multilayer_neural_network(train_subset=45000, valid_size=5000, test=False):
  train_dataset, train_labels, valid_dataset, valid_labels = \
      get_train_valid_data(train_subset, valid_size)


if __name__ == '__main__':
  run_multilayer_neural_network(train_subset=45000, valid_size=5000, test=False)
