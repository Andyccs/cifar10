from common_functions import get_train_valid_data, reformat_dataset, accuracy, print_loss, print_accuracy, plot
from constants import num_labels, num_features
from images_to_matrices import label_matrices_to_csv
from images_to_matrices import load_test_data
from images_to_matrices import load_train_data
import numpy as np
import tensorflow as tf


def model(data, weights, biases):
  return tf.matmul(data, weights) + biases


def run_multinomial_logistic_regression(train_subset=45000, valid_size=5000, test=True):
  """
  In Multinomial Logistic Regression, we have 
  input X of (n X image_size * image_size * color_channel) dimension and
  output Y of (n X num_labels) dimension, and Y is defined as:

    Y = softmax( X * W + b )

  where W and b are weights and biases. The loss function is defined as:

    Loss = cross_entropy(Y, labels)

  We use stochastic gradient descent, with batch size of 128, learning rate of 0.5 and 3001 steps. 
  We do not use any regularization because it does not improve the accuracy for this case. 
  At the end of the training, accuracy curve, loss curve will be plotted.

  Take note that train_subset + valid_size cannot be more than 50000 and train_subset cannot be 
  less than 128 (the batch size)

  Keyword arguments:
    train_subset -- the number of training example
    valid_size -- number data in validation set
    test -- if true, output a .csv file that predict 300000 data in testing set
  """
  train_dataset, train_labels, valid_dataset, valid_labels = \
      get_train_valid_data(train_subset, valid_size)

  print 'Building graph...'
  batch_size = 128

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)

    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    train_logits = model(tf_train_dataset, weights, biases)
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
    train_prediction = tf.nn.softmax(train_logits)

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(train_loss)

    # Predictions for the training, validation, and test data.
    valid_logits = model(tf_valid_dataset, weights, biases)
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, tf_valid_labels))
    valid_prediction = tf.nn.softmax(valid_logits)

  print 'Training...'

  num_steps = 3001

  trained_weights = np.ndarray(shape=(num_features, num_labels))
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
      if step % 100 == 0:
        print('Complete %.2f %%' % (float(step) / num_steps * 100.0))

    # Plot losses and accuracies
    print_loss(train_losses[-1], valid_losses[-1])
    print_accuracy(train_accuracies[-1], valid_accuracies[-1])
    plot(train_losses, valid_losses, 'Iteration', 'Loss')
    plot(train_accuracies, valid_accuracies, 'Iteration', 'Accuracy')

  if not test:
    return train_losses[-1], valid_losses[-1]

  part_size = 50000

  test_graph = tf.Graph()
  with test_graph.as_default():
    tf_test_dataset = tf.placeholder(tf.float32, shape=(part_size, num_features))
    weights = tf.constant(trained_weights)
    biases = tf.constant(trained_biases)

    logits = model(tf_test_dataset, weights, biases)
    test_prediction = tf.nn.softmax(logits)

  test_dataset = load_test_data()
  test_dataset = reformat_dataset(test_dataset)
  total_part = 6

  test_predicted_labels = np.ndarray(shape=(300000, 10))

  for i in range(total_part):
    test_dataset_part = test_dataset[i * part_size:(i + 1) * part_size]
    with tf.Session(graph=test_graph) as session:
      tf.initialize_all_variables().run()
      feed_dict = {tf_test_dataset: test_dataset_part}
      predict = session.run([test_prediction], feed_dict=feed_dict)
      test_predicted_labels[i * part_size:(i + 1) * part_size, :] = np.asarray(predict)[0]

  test_predicted_labels = np.argmax(test_predicted_labels, 1)

  label_matrices_to_csv(test_predicted_labels, 'submission.csv')


def plot_learning_curve():
  """
  Plot the learning curve of multinomial logistic regression model. X-axis is the size of 
  training set and y-axis is the value of loss function. Green color is loss for training data 
  and red color is loss for validation data
  """
  t_loss = []
  v_loss = []

  for m in range(1, 19):
    print 'm: ', m
    t, v = run_multinomial_logistic_regression(train_subset=m * 2500, test=False)
    t_loss.append(t)
    v_loss.append(v)

  plot(t_loss, vloss, 'Training Size', 'Loss')


if __name__ == '__main__':
  run_multinomial_logistic_regression(train_subset=45000, valid_size=5000, test=False)
