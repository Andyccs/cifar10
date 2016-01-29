import matplotlib.pyplot as plt
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


def run_multinomial_logistic_regression(train_subset=50000, test=True):
  train_dataset, train_labels = load_train_data()
  train_dataset = reformat_dataset(train_dataset)
  train_labels = reformat_labels(train_labels)

  # Create a validation dataset
  valid_size = 5000

  valid_dataset = train_dataset[:valid_size, :]
  valid_labels = train_labels[:valid_size]
  train_dataset = train_dataset[valid_size:valid_size + train_subset, :]
  train_labels = train_labels[valid_size:valid_size + train_subset]
  print 'Training set size:', train_dataset.shape, train_labels.shape
  print 'Validation set size:', valid_dataset.shape, valid_labels.shape

  print 'Building graph...'
  batch_size = 128

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size * color_channel))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)

    weights = tf.Variable(tf.truncated_normal([image_size * image_size * color_channel, num_labels
                                              ]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Y = X * W + b
    logits = tf.matmul(tf_train_dataset, weights) + biases
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(train_loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    valid_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            tf.matmul(tf_valid_dataset, weights) + biases, tf_valid_labels))

  print 'Training...'

  num_steps = 3001

  trained_weights = np.ndarray(shape=(image_size * image_size * color_channel, num_labels))
  trained_biases = np.ndarray(shape=(num_labels))

  train_losses = []
  valid_losses = []

  train_accuracies = []
  valid_accuracies = []

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'

    for step in xrange(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

      _, tl, vl, predictions, trained_weights, trained_biases = session.run(
          [optimizer, train_loss, valid_loss, train_prediction, weights, biases], 
          feed_dict=feed_dict)

      train_losses.append(tl)
      valid_losses.append(vl)
      train_accuracies.append(accuracy(predictions, batch_labels))
      valid_accuracies.append(accuracy(valid_prediction.eval(), valid_labels))

    # Plot losses and accuracies
    print 'Training loss: ', train_losses[-1]
    print 'Valid loss at step:', valid_losses[-1]
    print 'Training accuracy: %.2f%%' % train_accuracies[-1]
    print 'Validation accuracy: %.2f%%' % valid_accuracies[-1]

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(train_losses)), train_losses, color='g', label='Train')
    plt.plot(range(len(train_losses)), valid_losses, color='r', label='Valid')
    plt.show()

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(range(len(train_accuracies)), train_accuracies, color='g', label='Train')
    plt.plot(range(len(valid_accuracies)), valid_accuracies, color='r', label='Valid')
    plt.show()

  if not test:
    return train_losses[-1], valid_losses[-1]

  part_size = 50000

  test_graph = tf.Graph()
  with test_graph.as_default():
    tf_test_dataset = tf.placeholder(tf.float32, shape=(part_size, image_size * image_size * color_channel))
    weights = tf.placeholder(tf.float32, shape=(image_size * image_size * color_channel, num_labels))
    biases = tf.placeholder(tf.float32, shape=(num_labels))

    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

  test_dataset = load_test_data()
  test_dataset = reformat_dataset(test_dataset)
  total_part = 6

  test_predicted_labels = np.ndarray(shape=(300000, 10))

  for i in range(total_part):
    with tf.Session(graph=test_graph) as session:
      tf.initialize_all_variables().run()

      test_dataset_part = test_dataset[i*part_size: (i+1) * part_size]
      feed_dict = {
          tf_test_dataset: test_dataset_part, 
          weights: trained_weights,
          biases: trained_biases
      }
      predict = session.run([test_prediction], feed_dict=feed_dict)
      test_predicted_labels[i*part_size: (i+1) * part_size, :] = np.asarray(predict)[0]

  test_predicted_labels = np.argmax(test_predicted_labels, 1)

  label_matrices_to_csv(test_predicted_labels, 'submission.csv')


def plot_learning_curve():
  t_loss = []
  v_loss = []

  for m in range(1,19):
    print 'm: ', m
    t, v = run_multinomial_logistic_regression(train_subset=m*2500, test=False)
    t_loss.append(t)
    v_loss.append(v)

  plt.xlabel('Training Size')
  plt.ylabel('Loss')
  plt.plot(range(len(t_loss)), t_loss, color='g', label='Train')
  plt.plot(range(len(v_loss)), v_loss, color='r', label='Valid')
  plt.show()


if __name__ == '__main__':
  run_multinomial_logistic_regression(train_subset=50000, test=False)

