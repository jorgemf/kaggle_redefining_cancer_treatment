import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from configuration import *
from text_classification_model_simple import ModelSimple
from text_classification_train import TextClassificationTrainer
from text_classification_dataset import TextClassificationDataset


class ModelSimpleCNN(ModelSimple):
    """
    Text classification using convolution layers before the stack of recurrent GRU cells
    """

    def model(self, input_text, num_output_classes, embeddings, num_hidden=TC_MODEL_HIDDEN,
              num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT, cnn_filters=TC_CNN_FILTERS,
              cnn_layers=TC_CNN_LAYERS, training=True):

        embedded_sequence, sequence_length = self.model_embedded_sequence(embeddings, input_text)
        batch_size, max_length, _ = tf.unstack(tf.shape(embedded_sequence))

        conv_sequence = embedded_sequence
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
        type = embedded_sequence.dtype

        sequence_output, _ = tf.nn.dynamic_rnn(network, conv_sequence, dtype=tf.float32,
                                               sequence_length=sequence_length,
                                               initial_state=network.zero_state(batch_size, type))
        # get last output of the dynamic_rnn
        sequence_output = tf.reshape(sequence_output, [batch_size * max_length, num_hidden])
        indexes = tf.range(batch_size) * max_length + (sequence_length - 1)
        output = tf.gather(sequence_output, indexes)

        # full connected layer
        output = tf.nn.dropout(output, dropout)
        logits = layers.fully_connected(output, num_output_classes, activation_fn=None)

        prediction = tf.nn.softmax(logits)

        return {
            'logits': logits,
            'prediction': prediction,
        }


if __name__ == '__main__':
    trainer = TextClassificationTrainer(dataset=TextClassificationDataset(type='train'),
                                        text_classification_model=ModelSimpleCNN(),
                                        logdir='{}_{}'.format(DIR_TC_LOGDIR, 'cnn'))
    trainer.train()
