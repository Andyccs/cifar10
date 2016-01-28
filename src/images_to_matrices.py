import os
import numpy as np
from scipy import ndimage
import cPickle as pickle

def images_to_matrices(data_dir):
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

    print('Complete %.2f %% for %s' % (float(i) / num_images * 100.0, data_dir) )

  return dataset

def csv_to_label_matrices(filename):
  class_name = {
      'airplane': 0,
      'automobile': 1,
      'bird': 2,
      'cat': 3,
      'deer': 4,
      'dog': 5,
      'frog': 6,
      'horse': 7,
      'ship': 9,
      'truck': 10
  }

  data = np.genfromtxt(filename, dtype=None, delimiter=',', skip_header=1) 
  labels = [ class_name[d[1]] for d in data ]
  labels = np.array(labels)
  return labels

if __name__ == '__main__':
  train_data_dir = 'traindata'
  train_label_file = 'dataset/trainLabels.csv'

  train_dataset = images_to_matrices(train_data_dir)
  train_labels = csv_to_label_matrices(train_label_file)

  test_data_dir = 'testdata'
  test_dataset = images_to_matrices(test_data_dir)

  ## Save all matrices to a pickle file so that we can use them later
  
  # Save train data and labels
  train_pickle_file = 'cifar10_train.pickle'

  try:
    train_file = open(train_pickle_file, 'wb')
    save = {
      'train_dataset': train_dataset,
      'train_labels': train_labels
      }
    pickle.dump(save, train_file, pickle.HIGHEST_PROTOCOL)
    train_file.close()
  except Exception as e:
    print 'Unable to save data to', train_pickle_file, ':', e
    raise

  # Save test data
  test_pickle_file = 'cifar10_test.pickle'

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