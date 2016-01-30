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

  batch_size = 128
  hidden_layer_unit_1 = 5000

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)

    layer1_weights = weight_variable([num_features, hidden_layer_unit_1])
    layer1_bias = bias_variable([hidden_layer_unit_1])

    layer2_weights = weight_variable([hidden_layer_unit_1, num_labels])
    layer2_biases = bias_variable([num_labels])

    def model(data):
      u1 = tf.matmul(data, layer1_weights) + layer1_bias
      y1 = tf.nn.relu(u1)
      u2 = tf.matmul(y1, layer2_weights) + layer2_biases
      return u2

    train_logits = model(tf_train_dataset)
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits,
                                                                        tf_train_labels))
    train_prediction = tf.nn.softmax(train_logits)

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(train_loss)

    valid_logits = model(tf_valid_dataset)
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,
                                                                        tf_valid_labels))
    valid_prediction = tf.nn.softmax(valid_logits)

  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []
  num_steps = 3001

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print 'Initialized'

    for step in xrange(num_steps):
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

      batch_data = train_dataset[offset:(offset + batch_size), :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

      _, tl, predictions = session.run(
          [optimizer, train_loss, train_prediction],
          feed_dict=feed_dict)

      train_losses.append(tl)
      valid_losses.append(valid_loss.eval())
      train_accuracies.append(accuracy(predictions, batch_labels))
      valid_accuracies.append(accuracy(valid_prediction.eval(), valid_labels))

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
