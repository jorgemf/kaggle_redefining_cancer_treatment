import tensorflow as tf
import tensorflow.contrib.layers as layers
from configuration import *
from text_classification_model_simple import ModelSimple
from text_classification_train import TextClassificationTrainer
from text_classification_dataset import TextClassificationDataset


class ModelHATT(ModelSimple):
    def model(self, input_words, num_output_classes, embeddings, num_hidden=TC_MODEL_HIDDEN,
              dropout=TC_MODEL_DROPOUT, word_output_size=TC_HATT_WORD_OUTPUT_SIZE,
              sentence_output_size=TC_HATT_SENTENCE_OUTPUT_SIZE, training=True):
        # input_words [document x sentence x word]
        embeddings_size = len(embeddings[0])

        embedded_sequence, sentences_length, words_length = \
            self.model_embedded_sequence(embeddings, input_words)
        batch_size, sentence_size, word_size, _ = tf.unstack(tf.shape(embedded_sequence))

        # RNN word level
        with tf.variable_scope('word_level'):
            word_level_inputs = tf.reshape(embedded_sequence,
                                           [batch_size * sentence_size, word_size, embeddings_size])
            word_level_lengths = tf.reshape(words_length, [batch_size * sentence_size])

            word_level_output = self._bidirectional_rnn(word_level_inputs, word_level_lengths,
                                                        num_hidden)
            word_level_output = self._attention(word_level_output, word_output_size)
            word_level_output = layers.dropout(word_level_output, keep_prob=dropout,
                                               is_training=training)
        # RNN sentence level
        with tf.variable_scope('sentence_level'):
            sentence_level_inputs = tf.reshape(word_level_output,
                                               [batch_size, sentence_size, word_output_size])
            sentence_level_output = self._bidirectional_rnn(sentence_level_inputs, sentences_length,
                                                            num_hidden)
            sentence_level_output = self._attention(sentence_level_output, sentence_output_size)
            sentence_level_output = layers.dropout(sentence_level_output, keep_prob=dropout,
                                                   is_training=training)

        # classifier

        logits = layers.fully_connected(sentence_level_output, num_output_classes,
                                        activation_fn=None)
        prediction = tf.nn.softmax(logits)

        return {
            'logits': logits,
            'prediction': prediction,
        }

    def model_embedded_sequence(self, embeddings, input_text):
        """
        Given the embeddings and the input text returns the embedded sequence and
        the sentence length and words length
        :param embeddings:
        :param input_text:
        :return: (embedded_sequence, sentences_length, words_length)
        """
        # calculate max length of the input_text
        mask_words = tf.greater(input_text, 0)  # true for words false for padding
        words_length = tf.reduce_sum(tf.cast(mask_words, tf.int32), -1)
        mask_sentences = tf.greater(words_length, 0)
        sentences_length = tf.reduce_sum(tf.cast(mask_sentences, tf.int32), 1)

        # create the embeddings
        # first vector is a zeros vector used for padding
        embeddings_dimension = len(embeddings[0])
        embeddings = [[0.0] * embeddings_dimension] + embeddings
        embeddings = tf.constant(embeddings, name='embeddings', dtype=tf.float32)
        # this means we need to add 1 to the input_text
        input_text = tf.add(input_text, 1)
        embedded_sequence = tf.nn.embedding_lookup(embeddings, input_text)
        return embedded_sequence, sentences_length, words_length

    def _bidirectional_rnn(self, inputs_embedding, inputs_length, num_hidden):
        # Recurrent network.
        batch_size, max_length, _ = tf.unstack(tf.shape(inputs_embedding))
        cell_fw = tf.nn.rnn_cell.GRUCell(num_hidden)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_hidden)
        type = inputs_embedding.dtype
        (fw_outputs, bw_outputs), _ = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            initial_state_fw=cell_fw.zero_state(batch_size, type),
                                            initial_state_bw=cell_bw.zero_state(batch_size, type),
                                            inputs=inputs_embedding,
                                            dtype=tf.float32,
                                            swap_memory=True,
                                            sequence_length=inputs_length)
        sequence_output = tf.concat((fw_outputs, bw_outputs), 2)
        return sequence_output

    def _attention(self, inputs, output_size, initializer=layers.xavier_initializer(),
                   activation_fn=tf.tanh):
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn)

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector),
                                    axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs


if __name__ == '__main__':
    dataset = TextClassificationDataset(type='train', sentence_split=True)
    trainer = TextClassificationTrainer(dataset=dataset, text_classification_model=ModelHATT(),
                                        logdir='{}_{}'.format(DIR_TC_LOGDIR, 'hatt'))
    trainer.train()
