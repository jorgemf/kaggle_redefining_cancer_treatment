import tensorflow as tf
import tensorflow.contrib.layers as layers
from src.configuration import *
from src.text_classification_model_simple import ModelSimple
from src.text_classification_train import main


class ModelSimpleBidirectional(ModelSimple):
    """
    Text classification using a bidirectional dynamic rnn and GRU cells
    """

    def model(self, input_text, num_output_classes, batch_size, embeddings,
              num_hidden=TC_MODEL_HIDDEN, num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT,
              training=True):
        embedded_sequence, sequence_length = self.model_embedded_sequence(embeddings, input_text)
        _, max_length, _ = tf.unstack(tf.shape(embedded_sequence))

        # Recurrent network.
        cell_fw = tf.nn.rnn_cell.GRUCell(num_hidden)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_hidden)
        type = embedded_sequence.dtype
        (fw_outputs, bw_outputs), _ = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            initial_state_fw=cell_fw.zero_state(batch_size, type),
                                            initial_state_bw=cell_bw.zero_state(batch_size, type),
                                            inputs=embedded_sequence,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            sequence_length=sequence_length)
        sequence_output = tf.concat((fw_outputs, bw_outputs), 2)
        # get last output of the dynamic_rnn
        sequence_output = tf.reshape(sequence_output, [batch_size * max_length, num_hidden * 2])
        indexes = tf.range(batch_size) * max_length + (sequence_length - 1)
        output = tf.gather(sequence_output, indexes)

        # full connected layer
        output = layers.dropout(output, keep_prob=dropout, is_training=training)
        logits = layers.fully_connected(output, num_output_classes, activation_fn=None)

        prediction = tf.nn.softmax(logits)

        return {
            'logits': logits,
            'prediction': prediction,
        }


if __name__ == '__main__':
    main(ModelSimpleBidirectional(), 'bidirectional')
