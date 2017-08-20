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
              num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT):
        """
        Creates a model for text classification
        :param tf.Tensor input_text: the input data, the text as
        [batch_size, text_vector_max_length, embeddings_size]
        :param int num_output_classes: the number of output classes for the classifier
        :param List[List[float]] embeddings: a matrix with the embeddings for the embedding lookup
        :param int num_hidden: number of hidden GRU cells in every layer
        :param int num_layers: number of layers of the model
        :param float dropout: dropout value between layers
        :return Dict[str,tf.Tensor]: a dict with logits and prediction tensors
        """
        # first vector is a zeros vector used for padding
        embeddings_dimension = len(embeddings[0])
        embeddings = [[0.0] * embeddings_dimension] + embeddings
        embeddings = tf.Variable(embeddings, name='embeddings', dtype=tf.float32)
        # this means we need to add 1 to the input_text
        input_text = tf.add(input_text, 1)

        # mask to know where there is a word and where padding
        mask = tf.greater(input_text, 0)  # true for words false for padding
        # length of the sequences without padding
        sequence_length = tf.reduce_sum(tf.cast(mask, tf.int32), 1)

        embedded_sequence = tf.nn.embedding_lookup(embeddings, input_text)
        # TODO convolution over sequence?

        # Recurrent network.
        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.GRUCell(num_hidden)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(cells)
        output, _ = tf.nn.dynamic_rnn(network, embedded_sequence, dtype=tf.float32,
                                      sequence_length=sequence_length)
        # TODO tf.nn.bidirectional_rnn
        # output is [max_time, batch_size, cell.output_size]
        # Flatten to apply same weights to all time steps
        output = tf.reshape(output, [-1, num_hidden])
        tf.summary.histogram('dynamic_rnn_output', output)
        output = slim.batch_norm(output, scope='output_batch_norm')

        weight = tf.truncated_normal([num_hidden, num_output_classes], stddev=0.01)
        bias = tf.constant(0.1, shape=[num_output_classes])
        logits = tf.matmul(output, weight) + bias
        tf.summary.histogram('dynamic_rnn_logits', output)
        prediction = tf.nn.softmax(logits)
        dim_sequence_length = int(input_text.get_shape()[1])
        logits = tf.reshape(logits, [-1, dim_sequence_length, num_output_classes])
        prediction = tf.reshape(prediction, [-1, dim_sequence_length, num_output_classes])

        # calculate final prediction based on the average of the predictions in each step
        reshaped_mask = tf.reshape(tf.cast(mask, dtype=tf.float32), [-1, dim_sequence_length, 1])
        prediction_mask = prediction * reshaped_mask
        prediction_sum = tf.reduce_sum(prediction_mask, reduction_indices=1)
        reshaped_length = tf.reshape(tf.cast(sequence_length, tf.float32), [-1, 1])
        prediction_mean = prediction_sum / reshaped_length
        prediction_mean_sum = tf.reduce_sum(prediction_mean, reduction_indices=1)
        prediction_mean_sum = tf.reshape(prediction_mean_sum, [-1, 1])
        prediction_mean = prediction_mean / prediction_mean_sum

        prediction_classes = tf.argmax(prediction, axis=2)
        prediction_classes_one_hot = tf.one_hot(prediction_classes, axis=-1,
                                                depth=num_output_classes,
                                                on_value=1, off_value=0)
        prediction_classes_one_hot *= tf.reshape(tf.cast(mask, dtype=tf.int32),
                                                 [-1, dim_sequence_length, 1])
        prediction_classes_one_hot_sum = tf.reduce_sum(prediction_classes_one_hot,
                                                       reduction_indices=1)
        prediction_percent = tf.cast(prediction_classes_one_hot_sum, tf.float32) / reshaped_length

        return {
            'logits': logits,
            'prediction': prediction,
            'prediction_mean': prediction_mean,
            'prediction_percent': prediction_percent,
            'mask': mask,
            'sequence_length': sequence_length,
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
        mask = graph_data['mask']
        sequence_length = graph_data['sequence_length']
        # calculate numerical stable cross entropy
        max_logits = tf.reduce_max(logits, axis=1)
        num_output_classes = int(max_logits.get_shape()[1])
        max_logits = tf.reshape(max_logits, [-1, 1, num_output_classes])
        log_exp_logits = tf.log(tf.exp(logits))
        exp_logits_max = tf.exp(logits - max_logits)
        log_sum_exp_logits_max = tf.log(tf.reduce_sum(exp_logits_max, axis=2))
        size_sequences = int(log_sum_exp_logits_max.get_shape()[1])
        log_sum_exp_logits_max = tf.reshape(log_sum_exp_logits_max, [-1, size_sequences, 1])

        # Compute cross entropy for each frame.
        cross_entropy = targets * (log_exp_logits - max_logits - log_sum_exp_logits_max)
        cross_entropy = - tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= tf.cast(mask, dtype=tf.float32)
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(sequence_length, tf.float32)
        return tf.reduce_mean(cross_entropy)

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
        optimizer = tf.train.AdamOptimizer(learning_rate)
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
