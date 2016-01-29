from common_functions import reformat_dataset, reformat_labels
from constants import image_size, color_channel, num_labels
from images_to_matrices import label_matrices_to_csv
from images_to_matrices import load_test_data
from images_to_matrices import load_train_data
import numpy as np
import tensorflow as tf


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def build_graph_and_run(test_dataset, weights, biases):
  test_graph = tf.Graph()
  with test_graph.as_default():
    tf_test_dataset_1 = tf.constant(test_dataset)
    tf_weights = tf.constant(weights)
    tf_biases = tf.constant(biases)

    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset_1, weights) + biases)

  with tf.Session(graph=test_graph) as session:
    tf.initialize_all_variables().run()
    predict = session.run([test_prediction])
    return np.asarray(predict)[0]


def run_multinomial_logistic_regression(train_subset=45000):
  train_dataset, train_labels = load_train_data()
  train_dataset = reformat_dataset(train_dataset)
  train_labels = reformat_labels(train_labels)

  test_dataset = load_test_data()
  test_dataset = reformat_dataset(test_dataset)

  # Create a validation dataset
  valid_size = 5000

  valid_dataset = train_dataset[:valid_size, :]
  valid_labels = train_labels[:valid_size]
  train_dataset = train_dataset[valid_size:valid_size + train_subset, :]
  train_labels = train_labels[valid_size:valid_size + train_subset]
  print 'Training set size:', train_dataset.shape, train_labels.shape
  print 'Validation set size:', valid_dataset.shape, valid_labels.shape

  print 'Building graph...'
  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset)
    tf_train_labels = tf.constant(train_labels)
    tf_valid_dataset = tf.constant(valid_dataset)

    weights = tf.Variable(tf.truncated_normal([image_size * image_size * color_channel, num_labels
                                              ]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Y = X * W + b
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)

  print 'Training...'

  num_steps = 801

  trained_weights = np.ndarray(shape=(image_size * image_size * color_channel, num_labels))
  trained_biases = np.ndarray(shape=(num_labels))

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'

    for step in xrange(num_steps):
      _, l, predictions, trained_weights, trained_biases = session.run(
          [optimizer, loss, train_prediction, weights, biases])
      if (step % 100 == 0):
        print 'Loss at step', step, ':', l
        print 'Training accuracy: %.2f%%' % accuracy(predictions, train_labels)
        print 'Validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), valid_labels)

  # Separate test dataset into 6 parts, because a tensor graph cannot be more than 2GB
  test_predicted_labels_1 = build_graph_and_run(test_dataset[0:50000], trained_weights,
                                                trained_biases)
  test_predicted_labels_2 = build_graph_and_run(test_dataset[50000:100000], trained_weights,
                                                trained_biases)
  test_predicted_labels_3 = build_graph_and_run(test_dataset[100000:150000], trained_weights,
                                                trained_biases)
  test_predicted_labels_4 = build_graph_and_run(test_dataset[150000:200000], trained_weights,
                                                trained_biases)
  test_predicted_labels_5 = build_graph_and_run(test_dataset[200000:250000], trained_weights,
                                                trained_biases)
  test_predicted_labels_6 = build_graph_and_run(test_dataset[250000:300000], trained_weights,
                                                trained_biases)

  test_predicted_labels = np.concatenate(
      (test_predicted_labels_1, test_predicted_labels_2, test_predicted_labels_3,
       test_predicted_labels_4, test_predicted_labels_5, test_predicted_labels_6))
  test_predicted_labels = np.argmax(test_predicted_labels, 1)

  label_matrices_to_csv(test_predicted_labels, 'submission.csv')


if __name__ == '__main__':
  run_multinomial_logistic_regression(train_subset=45000)
