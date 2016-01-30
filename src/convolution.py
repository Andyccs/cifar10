from common_functions import get_train_valid_data, accuracy
from constants import image_size, color_channel, num_labels
import tensorflow as tf


def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
  return tf.Variable(tf.zeros(shape))


def constant_bias_variable(constant, shape):
  return tf.Variable(tf.constant(constant, shape=shape))


def run_convolution(train_subset=45000, valid_size=5000, test=False):
  train_dataset, train_labels, valid_dataset, valid_labels = \
      get_train_valid_data(train_subset, valid_size, reformat_datag=False)

  print 'Building graph...'

  batch_size = 16
  patch_size = 5
  depth = 16
  num_hidden = 64

  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, color_channel))
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

    # Model.
    def model(data):
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

    train_logits = model(tf_train_dataset)
    train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
    train_prediction = tf.nn.softmax(train_logits)

    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(train_loss)

    valid_logits = model(tf_valid_dataset)
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, tf_valid_labels))
    valid_prediction = tf.nn.softmax(valid_logits)


if __name__ == '__main__':
  run_convolution(train_subset=45000, valid_size=5000, test=False)
