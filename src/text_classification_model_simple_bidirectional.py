import tensorflow as tf
from configuration import *
from text_classification_model_simple import ModelSimple


class ModelSimpleBidirectional(ModelSimple):

    def model(self, input_text, num_output_classes, embeddings, num_hidden=TC_MODEL_HIDDEN,
              num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT, training=True):

        embedded_sequence, sequence_length = self.model_embedded_sequence(embeddings, input_text)

        # Recurrent network.
        cell_fw = tf.nn.rnn_cell.GRUCell(num_hidden)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_hidden)
        batch_size = tf.shape(input_text)[0]
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

        output = self.model_sequence_output(sequence_output, sequence_length)

        logits, prediction = self.model_full_connected_logits_prediction(output, num_hidden*2,
                                                                         num_output_classes)
        return {
            'logits': logits,
            'prediction': prediction,
        }
