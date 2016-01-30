from common_functions import get_train_valid_data, accuracy, print_loss, print_accuracy, plot
from constants import num_features, num_labels
import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.zeros(shape)
  return tf.Variable(initial)


def run_multilayer_neural_network(train_subset=45000, valid_size=5000, test=False):
  train_dataset, train_labels, valid_dataset, valid_labels = \
      get_train_valid_data(train_subset, valid_size)

  print 'Building graph...'

  hidden_layer_unit_1 = 5000

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    w1 = weight_variable([num_features, hidden_layer_unit_1])
    b1 = bias_variable([hidden_layer_unit_1])
    u1 = tf.matmul(tf_train_dataset, w1) + b1

    y1 = tf.nn.relu(u1)
    w2 = weight_variable([hidden_layer_unit_1, num_labels])
    b2 = bias_variable([num_labels])
    u2 = tf.matmul(y1, w2) + b2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(u2, tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(u2)

  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []

  num_steps = 3001
  batch_size = 128

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'

    for step in xrange(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

      _, tl, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

      train_losses.append(tl)
      train_accuracies.append(accuracy(predictions, batch_labels))

      validation_feed_dict = {tf_train_dataset: valid_dataset, tf_train_labels: valid_labels}
      vl = loss.eval(feed_dict=validation_feed_dict)
      valid_prediction = train_prediction.eval(feed_dict=validation_feed_dict)

      valid_losses.append(vl)
      valid_accuracies.append(accuracy(valid_prediction, valid_labels))

      if step % 100 == 0:
        print('Complete %.2f %%' % (float(step) / num_steps * 100.0))

    # Plot losses and accuracies
    print_loss(train_losses[-1], valid_losses[-1])
    print_accuracy(train_accuracies[-1], valid_accuracies[-1])
    plot(train_losses, valid_losses, 'Iteration', 'Loss')
    plot(train_accuracies, valid_accuracies, 'Iteration', 'Accuracy')

  # TODO: Generate submission.csv file


if __name__ == '__main__':
  run_multilayer_neural_network(train_subset=45000, valid_size=5000, test=False)
