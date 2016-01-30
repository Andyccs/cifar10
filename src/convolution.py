from common_functions import get_train_valid_data, accuracy, print_loss, print_accuracy, plot
from constants import image_size, color_channel, num_labels
import tensorflow as tf


def model(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, layer3_weights,
          layer3_biases, layer4_weights, layer4_biases):
  # Convolution layers
  conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
  hidden = tf.nn.relu(conv + layer1_biases)
  conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
  hidden = tf.nn.relu(conv + layer2_biases)

  # MLP layers
  shape = hidden.get_shape().as_list()
  reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
  hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
  result = tf.matmul(hidden, layer4_weights) + layer4_biases
  return result


def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))


def constant_bias_variable(constant, shape):
  return tf.Variable(tf.constant(constant, shape=shape))


def run_convolution(train_subset=45000, valid_size=5000, test=False):
  train_dataset, train_labels, valid_dataset, valid_labels = \
      get_train_valid_data(train_subset, valid_size, reformat_data=False)

  print 'Building graph...'
  batch_size = 16
  patch_size = 5
  depth = 16
  num_hidden = 64

  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size, image_size, color_channel))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)

    # Variables.
    layer1_weights = weight_variable([patch_size, patch_size, color_channel, depth])
    layer1_biases = bias_variable(shape=[depth])
    layer2_weights = weight_variable([patch_size, patch_size, depth, depth])
    layer2_biases = constant_bias_variable(1.0, [depth])
    layer3_weights = weight_variable([image_size / 4 * image_size / 4 * depth, num_hidden])
    layer3_biases = constant_bias_variable(1.0, [num_hidden])
    layer4_weights = weight_variable([num_hidden, num_labels])
    layer4_biases = constant_bias_variable(1.0, [num_labels])

    train_logits = model(tf_train_dataset, layer1_weights, layer1_biases, layer2_weights,
                         layer2_biases, layer3_weights, layer3_biases, layer4_weights,
                         layer4_biases)
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits,
                                                                        tf_train_labels))
    train_prediction = tf.nn.softmax(train_logits)

    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(train_loss)

    valid_logits = model(tf_valid_dataset, layer1_weights, layer1_biases, layer2_weights,
                         layer2_biases, layer3_weights, layer3_biases, layer4_weights,
                         layer4_biases)
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits,
                                                                        tf_valid_labels))
    valid_prediction = tf.nn.softmax(valid_logits)

  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []
  num_steps = 1001

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
  run_convolution(train_subset=45000, valid_size=5000, test=False)
