import tensorflow as tf
from tensorflow.contrib import slim
import tensorflow.contrib.layers as layers
from ..configuration import *
from .text_classification_train import main


class ModelSimple(object):
    """
    Base class to create models for text classification. It uses several layers of GRU cells.
    """

    def model(self, input_text_begin, input_text_end, gene, variation, num_output_classes,
              batch_size, embeddings, training=True, dropout=TC_MODEL_DROPOUT):

        """
        Creates a model for text classification
        :param tf.Tensor input_text: the input data, the text as
        [batch_size, text_vector_max_length, embeddings_size]
        :param int num_output_classes: the number of output classes for the classifier
        :param int batch_size: batch size, the same used in the dataset
        :param List[List[float]] embeddings: a matrix with the embeddings for the embedding lookup
        :param int num_hidden: number of hidden GRU cells in every layer
        :param int num_layers: number of layers of the model
        :param float dropout: dropout value between layers
        :param boolean training: whether the model is built for training or not
        :return Dict[str,tf.Tensor]: a dict with logits and prediction tensors
        """
        input_text_begin = tf.reshape(input_text_begin, [batch_size, MAX_WORDS])
        if input_text_end is not None:
            input_text_end = tf.reshape(input_text_end, [batch_size, MAX_WORDS])
        embedded_sequence_begin, sequence_length_begin, \
        embedded_sequence_end, sequence_length_end, \
        gene, variation = \
            self.model_embedded_sequence(embeddings, input_text_begin, input_text_end, gene,
                                         variation)
        _, max_length, _ = tf.unstack(tf.shape(embedded_sequence_begin))

        with tf.variable_scope('text_begin'):
            output_begin = self.rnn(embedded_sequence_begin, sequence_length_begin, max_length,
                                    dropout, batch_size, training)
        if input_text_end is not None:
            with tf.variable_scope('text_end'):
                output_end = self.rnn(embedded_sequence_end, sequence_length_end, max_length,
                                      dropout, batch_size, training)
            output = tf.concat([output_begin, output_end], axis=1)
        else:
            output = output_begin

        # full connected layer
        logits = self.model_fully_connected(output, gene, variation, num_output_classes, dropout,
                                            training)

        prediction = tf.nn.softmax(logits)

        return {
            'logits'    : logits,
            'prediction': prediction,
            }

    def rnn(self, sequence, sequence_length, max_length, dropout, batch_size, training,
            num_hidden=TC_MODEL_HIDDEN, num_layers=TC_MODEL_LAYERS):
        # Recurrent network.
        cells = []
        for _ in range(num_layers):
            cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            if training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(cells)
        type = sequence.dtype

        sequence_output, _ = tf.nn.dynamic_rnn(network, sequence, dtype=tf.float32,
                                               sequence_length=sequence_length,
                                               initial_state=network.zero_state(batch_size, type))
        # get last output of the dynamic_rnn
        sequence_output = tf.reshape(sequence_output, [batch_size * max_length, num_hidden])
        indexes = tf.range(batch_size) * max_length + (sequence_length - 1)
        output = tf.gather(sequence_output, indexes)
        return output

    def model_fully_connected(self, output, gene, variation, num_output_classes, dropout, training):
        output = layers.dropout(output, keep_prob=dropout, is_training=training)
        net = tf.concat([output, gene, variation], axis=1)
        net = layers.fully_connected(net, 128, activation_fn=tf.nn.relu)
        net = layers.dropout(net, keep_prob=dropout, is_training=training)
        logits = layers.fully_connected(net, num_output_classes, activation_fn=None)
        return logits

    def remove_padding(self, input_text):
        # calculate max length of the input_text
        mask = tf.greater_equal(input_text, 0)  # true for words false for padding
        sequence_length = tf.reduce_sum(tf.cast(mask, tf.int32), 1)

        # truncate the input text to max length
        max_sequence_length = tf.reduce_max(sequence_length)
        input_text_length = tf.shape(input_text)[1]
        empty_padding_lenght = input_text_length - max_sequence_length
        input_text, _ = tf.split(input_text, [max_sequence_length, empty_padding_lenght], axis=1)

        return input_text, sequence_length

    def model_embedded_sequence(self, embeddings, input_text_begin, input_text_end, gene,
                                variation):
        """
        Given the embeddings and the input text returns the embedded sequence and
        the sequence length. The input_text is truncated to the max length of the sequence, so
        the output embedded_sequence wont have the same shape as input_text or even a constant shape
        :param embeddings:
        :param input_text:
        :return: (embedded_sequence, sequence_length)
        """
        input_text_begin, sequence_length_begin = self.remove_padding(input_text_begin)
        if input_text_end is not None:
            input_text_end, sequence_length_end = self.remove_padding(input_text_end)
        else:
            sequence_length_end = None
        variation, variation_length = self.remove_padding(variation)

        # create the embeddings

        # first vector is a zeros vector used for padding
        embeddings_dimension = len(embeddings[0])
        embeddings = [[0.0] * embeddings_dimension] + embeddings
        embeddings = tf.constant(embeddings, name='embeddings', dtype=tf.float32)
        # this means we need to add 1 to the input_text
        input_text_begin = tf.add(input_text_begin, 1)
        if input_text_end is not None:
            input_text_end = tf.add(input_text_end, 1)
        gene = tf.add(gene, 1)
        variation = tf.add(variation, 1)
        embedded_sequence_begin = tf.nn.embedding_lookup(embeddings, input_text_begin)
        if input_text_end is not None:
            embedded_sequence_end = tf.nn.embedding_lookup(embeddings, input_text_end)
        else:
            embedded_sequence_end = None
        embedded_gene = tf.nn.embedding_lookup(embeddings, gene)
        embedded_gene = tf.squeeze(embedded_gene, axis=1)
        embedded_variation = tf.nn.embedding_lookup(embeddings, variation)
        embedded_variation = tf.reduce_mean(embedded_variation, axis=1)

        return embedded_sequence_begin, sequence_length_begin, \
               embedded_sequence_end, sequence_length_end, \
               embedded_gene, embedded_variation

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
        targets = tf.squeeze(targets, axis=1)
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
        # optimizer
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = optimizer.minimize(loss, global_step=global_step)
        return optimizer, learning_rate


if __name__ == '__main__':
    main(ModelSimple(), 'simple', batch_size=TC_BATCH_SIZE)
