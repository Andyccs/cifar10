from scipy import ndimage
import cPickle as pickle
import numpy as np
import os


def images_to_matrices(data_dir):
  """
  Convert all images in data_dir to a four dimensional matrix, with the first dimension as 
  the total number of images in data_dir, (32 X 32 X 3) for the last three dimension as the width,
  height and number of colour channel of an image
  """
  image_files = os.listdir(data_dir)

  num_images = len(image_files)
  image_size = 32
  pixel_depth = 255.0
  color_channel = 3

  dataset_shape = (num_images, image_size, image_size, color_channel)
  dataset = np.ndarray(shape=dataset_shape, dtype=np.float32)

  for i in range(num_images):
    try:
      image_file = os.path.join(data_dir, image_files[i])

      # Normalize the image pixel to roughly mean of 0 and standard deviation of 0.5
      image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

      # Check the image size
      if image_data.shape != (image_size, image_size, color_channel):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))

      dataset[i, :, :, :] = image_data

    except IOError as e:
      print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'

    print('Complete %.2f %% for %s' % (float(i) / num_images * 100.0, data_dir))

  return dataset


class_name = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}


def csv_to_label_matrices(filename):
  """
  Convert a csv file to an one dimensional array. Each element in the array has an integer value of
  range [0, 10], which represents 11 different classes as define in the 
  images_to_matrices.class_name variable
  """
  data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1)
  labels = [class_name[d[1]] for d in data]
  labels = np.array(labels)
  return labels


class_value = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
               'truck']


def label_matrices_to_csv(labels, filename):
  """
  Convert a one dimentional array of integer value to a csv file with (index, class_name) pair in
  each row
  """
  result = []
  for i in range(labels.shape[0]):
    index = i + 1
    label = labels[i]
    label_string = class_value[label]
    result.append([index, label_string])

  result = np.asarray(result)
  np.savetxt(filename, result, fmt='%s', delimiter=",", header='id,label', comments='')


train_pickle_file = 'cifar10_train.pickle'
test_pickle_file = 'cifar10_test.pickle'


def extract_data():
  """
  Convert all images in traindata and testdata folder and dataset/trainLabel.csv file to matrices,
  then store them in two pickle files
  """
  train_data_dir = 'traindata'
  train_label_file = 'dataset/trainLabels.csv'

  train_dataset = images_to_matrices(train_data_dir)
  train_labels = csv_to_label_matrices(train_label_file)

  test_data_dir = 'testdata'
  test_dataset = images_to_matrices(test_data_dir)

  # Save train data and labels
  try:
    train_file = open(train_pickle_file, 'wb')
    save = {'train_dataset': train_dataset, 'train_labels': train_labels}
    pickle.dump(save, train_file, pickle.HIGHEST_PROTOCOL)
    train_file.close()
  except Exception as e:
    print 'Unable to save data to', train_pickle_file, ':', e
    raise

  # Save test data
  # If we save all 300,000 test data as one part, there will be an unknown error
  # We save test data as 6 parts in the pickle file
  test_dataset_1 = test_dataset[0:50000]
  test_dataset_2 = test_dataset[50000:100000]
  test_dataset_3 = test_dataset[100000:150000]
  test_dataset_4 = test_dataset[150000:200000]
  test_dataset_5 = test_dataset[200000:250000]
  test_dataset_6 = test_dataset[250000:300000]

  try:
    test_file = open(test_pickle_file, 'wb')
    save = {
        'test_dataset_1': test_dataset_1,
        'test_dataset_2': test_dataset_2,
        'test_dataset_3': test_dataset_3,
        'test_dataset_4': test_dataset_4,
        'test_dataset_5': test_dataset_5,
        'test_dataset_6': test_dataset_6
    }
    pickle.dump(save, test_file, pickle.HIGHEST_PROTOCOL)
    test_file.close()
  except Exception as e:
    print 'Unable to save data to', test_pickle_file, ':', e
    raise


def load_train_data():
  """
  Load all matrices for training data from pickle file that we have saved 
  using extract_data() function
  """
  with open(train_pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    del save  # hint to help gc free up memory
    return train_dataset, train_labels


def load_test_data():
  """
  Load all matrices for testing data from pickle file that we have saved 
  using extract_data() function
  """
  with open(test_pickle_file, 'rb') as f:
    save = pickle.load(f)
    test_dataset_1 = save['test_dataset_1']
    test_dataset_2 = save['test_dataset_2']
    test_dataset_3 = save['test_dataset_3']
    test_dataset_4 = save['test_dataset_4']
    test_dataset_5 = save['test_dataset_5']
    test_dataset_6 = save['test_dataset_6']
    del save
    test_dataset = np.concatenate((test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4,
                                   test_dataset_5, test_dataset_6))
    return test_dataset


def load_data():
  """
  Load all matrices for training and testing data data from pickle file
  """
  train_dataset, train_labels = load_train_data()
  test_dataset = load_test_data()
  return train_dataset, train_labels, test_dataset


if __name__ == '__main__':
  extract_data()
