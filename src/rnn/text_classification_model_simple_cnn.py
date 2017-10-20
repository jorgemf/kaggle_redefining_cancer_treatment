import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from ..configuration import *
from .text_classification_model_simple import ModelSimple
from .text_classification_train import main


class ModelSimpleCNN(ModelSimple):
    """
    Text classification using convolution layers before the stack of recurrent GRU cells
    """

    def rnn(self, sequence, sequence_length, max_length, dropout, batch_size, training,
            num_hidden=TC_MODEL_HIDDEN, num_layers=TC_MODEL_LAYERS,
            cnn_filters=TC_CNN_FILTERS, cnn_layers=TC_CNN_LAYERS):

        conv_sequence = sequence
        for _ in range(cnn_layers):
            conv_sequence = layers.convolution(conv_sequence, cnn_filters, [5], activation_fn=None)

        # Recurrent network.
        cells = []
        for _ in range(num_layers):
            cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            if training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(cells)
        type = sequence.dtype

        sequence_output, _ = tf.nn.dynamic_rnn(network, conv_sequence, dtype=tf.float32,
                                               sequence_length=sequence_length,
                                               initial_state=network.zero_state(batch_size, type))
        # get last output of the dynamic_rnn
        sequence_output = tf.reshape(sequence_output, [batch_size * max_length, num_hidden])
        indexes = tf.range(batch_size) * max_length + (sequence_length - 1)
        output = tf.gather(sequence_output, indexes)
        return output


if __name__ == '__main__':
    main(ModelSimpleCNN(), 'cnn', batch_size=TC_BATCH_SIZE_CNN)
