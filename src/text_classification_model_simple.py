import tensorflow as tf
from configuration import *
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from configuration import *


class ModelSimple(object):
    """
    Base class to create models for text classification.
    """

    def model(self, input_text, num_output_classes, embeddings, num_hidden=TC_MODEL_HIDDEN,
              num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT, training=True):
        """
        Creates a model for text classification
        :param tf.Tensor input_text: the input data, the text as
        [batch_size, text_vector_max_length, embeddings_size]
        :param int num_output_classes: the number of output classes for the classifier
        :param List[List[float]] embeddings: a matrix with the embeddings for the embedding lookup
        :param int num_hidden: number of hidden GRU cells in every layer
        :param int num_layers: number of layers of the model
        :param float dropout: dropout value between layers
        :param boolean training: whether the model is built for training or not
        :return Dict[str,tf.Tensor]: a dict with logits and prediction tensors
        """
        # first vector is a zeros vector used for padding
        embeddings_dimension = len(embeddings[0])
        embeddings = [[0.0] * embeddings_dimension] + embeddings
        embeddings = tf.constant(embeddings, name='embeddings', dtype=tf.float32)
        # this means we need to add 1 to the input_text
        input_text = tf.add(input_text, 1)

        # mask to know where there is a word and where padding
        mask = tf.greater(input_text, 0)  # true for words false for padding
        # length of the sequences without padding
        sequence_length = tf.reduce_sum(tf.cast(mask, tf.int32), 1)

        embedded_sequence = tf.nn.embedding_lookup(embeddings, input_text)

        # Recurrent network.
        cells = []
        for _ in range(num_layers):
            cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            if training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(cells)
        sequence_output, _ = tf.nn.dynamic_rnn(network, embedded_sequence, dtype=tf.float32,
                                               sequence_length=sequence_length)

        # get the last relevant output of the sequence
        batch_size = tf.shape(sequence_output)[0]
        max_length = int(sequence_output.get_shape()[1])
        output_size = int(sequence_output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(sequence_output, [-1, output_size])
        output = tf.gather(flat, index)
        tf.summary.histogram('dynamic_rnn_output', output)
        # apply batch normalization to speed up training time
        output = slim.batch_norm(output, scope='output_batch_norm')
        # add a full connected layer
        weight = tf.truncated_normal([num_hidden, num_output_classes], stddev=0.01)
        bias = tf.constant(0.1, shape=[num_output_classes])
        logits = tf.matmul(output, weight) + bias
        tf.summary.histogram('dynamic_rnn_logits', output)
        prediction = tf.nn.softmax(logits)

        return {
            'logits': logits,
            'prediction': prediction,
        }

    def model_arg_scope(self, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon,
                            activation_fn=None) as scope:
            return scope

    def targets(self, labels, output_classes):
        """
        Transform a vector of labels into a matrix of one hot encoding labels
        :param tf.Tensor labels: an array of labels with dimension [batch_size]
        :param int output_classes: the total number of output classes
        :return tf.Tensor: a tensorflow tensor
        """
        targets = tf.one_hot(labels, axis=-1, depth=output_classes, on_value=1.0, off_value=0.0)
        return targets

    def loss(self, targets, graph_data):
        """
        Calculates the softmax cross entropy loss
        :param tf.Tensor logits: logits output of the model
        :param tf.Tensor targets: targets with the one hot encoding labels
        :return tf.Tensor : a tensor with the loss value
        """
        logits = graph_data['logits']
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        return tf.reduce_mean(loss)

    def optimize(self, loss, global_step,
                 learning_rate_initial=TC_LEARNING_RATE_INITIAL,
                 learning_rate_decay=TC_LEARNING_RATE_DECAY,
                 learning_rate_decay_steps=TC_LEARNING_RATE_DECAY_STEPS):
        """
        Creates a learning rate and an optimizer for the loss
        :param tf.Tensor loss: the tensor with the loss of the model
        :param tf.Tensor global_step: the global step for training
        :param int learning_rate_initial: the initial learning rate
        :param int learning_rate_decay: the decay of the learning rate
        :param int learning_rate_decay_steps: the number of steps to decay the learning rate
        :return (tf.Tensor, tf.Tensor): a tuple with the optimizer and the learning rate
        """
        learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step,
                                                   learning_rate_decay_steps,
                                                   learning_rate_decay,
                                                   staircase=True, name='learning_rate')
        # optimizer and gradient clipping
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        with ops.name_scope('summarize_grads'):
            for grad, var in zip(gradients, variables):
                if grad is not None:
                    if isinstance(grad, ops.IndexedSlices):
                        grad_values = grad.values
                    else:
                        grad_values = grad
                    tf.summary.histogram(var.op.name + '/gradient', grad_values)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer = optimizer.apply_gradients(zip(gradients, variables),
                                              global_step=global_step)
        return optimizer, learning_rate
