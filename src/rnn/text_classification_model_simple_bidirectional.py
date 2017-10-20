import tensorflow as tf
from ..configuration import *
from .text_classification_model_simple import ModelSimple
from .text_classification_train import main


class ModelSimpleBidirectional(ModelSimple):
    """
    Text classification using a bidirectional dynamic rnn and GRU cells
    """

    def rnn(self, sequence, sequence_length, max_length, dropout, batch_size, training,
            num_hidden=TC_MODEL_HIDDEN, num_layers=TC_MODEL_LAYERS):

        # Recurrent network.
        cell_fw = tf.nn.rnn_cell.GRUCell(num_hidden)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_hidden)
        type = sequence.dtype
        (fw_outputs, bw_outputs), _ = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            initial_state_fw=cell_fw.zero_state(batch_size, type),
                                            initial_state_bw=cell_bw.zero_state(batch_size, type),
                                            inputs=sequence,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            sequence_length=sequence_length)
        sequence_output = tf.concat((fw_outputs, bw_outputs), 2)
        # get last output of the dynamic_rnn
        sequence_output = tf.reshape(sequence_output, [batch_size * max_length, num_hidden * 2])
        indexes = tf.range(batch_size) * max_length + (sequence_length - 1)
        output = tf.gather(sequence_output, indexes)
        return output

if __name__ == '__main__':
    main(ModelSimpleBidirectional(), 'bidirectional', batch_size=TC_BATCH_SIZE_BIDIRECTIONAL)
