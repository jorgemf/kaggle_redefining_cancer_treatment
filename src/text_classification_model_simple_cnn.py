import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from configuration import *
from text_classification_model_simple import ModelSimple


class ModelSimpleCNN(ModelSimple):

    def model(self, input_text, num_output_classes, embeddings, num_hidden=TC_MODEL_HIDDEN,
              num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT, training=True):

        embedded_sequence, sequence_length = self.model_embedded_sequence(embeddings, input_text)

        conv_sequence = layers.convolution(embedded_sequence, 128, [5], activation_fn=None)

        # Recurrent network.
        cells = []
        for _ in range(num_layers):
            cell = tf.nn.rnn_cell.GRUCell(num_hidden)
            if training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(cells)
        batch_size = tf.shape(input_text)[0]
        type = embedded_sequence.dtype
        sequence_output, _ = tf.nn.dynamic_rnn(network, conv_sequence, dtype=tf.float32,
                                               sequence_length=sequence_length,
                                               initial_state=network.zero_state(batch_size, type))

        output = self.model_sequence_output(sequence_output, sequence_length)

        logits, prediction = self.model_full_connected_logits_prediction(output, num_hidden,
                                                                         num_output_classes)
        return {
            'logits': logits,
            'prediction': prediction,
        }
