import tensorflow as tf
import tensorflow.contrib.layers as layers
from ..configuration import *
from .text_classification_model_simple import ModelSimple
from .text_classification_train import main


class ModelHAN(ModelSimple):
    def _create_embeddings(self, embeddings):
        embeddings_dimension = len(embeddings[0])
        embeddings = [[0.0] * embeddings_dimension] + embeddings
        embeddings = tf.constant(embeddings, name='embeddings', dtype=tf.float32)
        return embeddings

    def _embed_sequence_with_length(self, embeddings, input_text):
        # calculate max length of the input_text
        mask_words = tf.greater(input_text, 0)  # true for words false for padding
        words_length = tf.reduce_sum(tf.cast(mask_words, tf.int32), -1)
        mask_sentences = tf.greater(words_length, 0)
        sentences_length = tf.reduce_sum(tf.cast(mask_sentences, tf.int32), 1)
        input_text = tf.add(input_text, 1)
        embedded_sequence = tf.nn.embedding_lookup(embeddings, input_text)
        return embedded_sequence, sentences_length, words_length

    def _embed(self, embeddings, gene, variation):
        variation, variation_length = self.remove_padding(variation)
        variation = tf.add(variation, 1)
        embedded_variation = tf.nn.embedding_lookup(embeddings, variation)
        embedded_variation = tf.reduce_mean(embedded_variation, axis=1)
        gene = tf.add(gene, 1)
        embedded_gene = tf.nn.embedding_lookup(embeddings, gene)
        embedded_gene = tf.squeeze(embedded_gene, axis=1)
        return embedded_gene, embedded_variation

    def _han(self, input_words, embeddings, gene, variation, batch_size, embeddings_size,
             num_hidden, dropout, word_output_size, sentence_output_size, training=True):

        input_words = tf.reshape(input_words, [batch_size, MAX_SENTENCES, MAX_WORDS_IN_SENTENCE])
        embedded_sequence, sentences_length, words_length = \
            self._embed_sequence_with_length(embeddings, input_words)
        _, sentence_size, word_size, _ = tf.unstack(tf.shape(embedded_sequence))

        # RNN word level
        with tf.variable_scope('word_level'):
            word_level_inputs = tf.reshape(embedded_sequence,
                                           [batch_size * sentence_size, word_size, embeddings_size])
            word_level_lengths = tf.reshape(words_length, [batch_size * sentence_size])

            word_level_output = self._bidirectional_rnn(word_level_inputs, word_level_lengths,
                                                        num_hidden)
            word_level_output = tf.reshape(word_level_output, [batch_size, sentence_size, word_size,
                                                               num_hidden * 2])
            word_level_output = self._attention(word_level_output, word_output_size, gene,
                                                variation)
            word_level_output = layers.dropout(word_level_output, keep_prob=dropout,
                                               is_training=training)
        # RNN sentence level
        with tf.variable_scope('sentence_level'):
            sentence_level_inputs = tf.reshape(word_level_output,
                                               [batch_size, sentence_size, word_output_size])
            sentence_level_output = self._bidirectional_rnn(sentence_level_inputs, sentences_length,
                                                            num_hidden)
            sentence_level_output = self._attention(sentence_level_output, sentence_output_size,
                                                    gene, variation)
            sentence_level_output = layers.dropout(sentence_level_output, keep_prob=dropout,
                                                   is_training=training)

        return sentence_level_output

    def model(self, input_text_begin, input_text_end, gene, variation,
              num_output_classes, batch_size, embeddings,
              num_hidden=TC_MODEL_HIDDEN, dropout=TC_MODEL_DROPOUT,
              word_output_size=TC_HATT_WORD_OUTPUT_SIZE,
              sentence_output_size=TC_HATT_SENTENCE_OUTPUT_SIZE, training=True):
        embeddings_size = len(embeddings[0])
        embeddings = self._create_embeddings(embeddings)
        gene, variation = self._embed(embeddings, gene, variation)

        with tf.variable_scope('text_begin'):
            hatt_begin = self._han(input_text_begin, embeddings, gene, variation, batch_size,
                                   embeddings_size, num_hidden, dropout, word_output_size,
                                   sentence_output_size, training)

        if input_text_end is not None:
            with tf.variable_scope('text_end'):
                hatt_end = self._han(input_text_end, embeddings, gene, variation, batch_size,
                                     embeddings_size, num_hidden, dropout, word_output_size,
                                     sentence_output_size, training)

            hatt = tf.concat([hatt_begin, hatt_end], axis=1)
        else:
            hatt = hatt_begin

        # classifier
        logits = layers.fully_connected(hatt, num_output_classes, activation_fn=None)
        prediction = tf.nn.softmax(logits)

        return {
            'logits'    : logits,
            'prediction': prediction,
            }

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

    def _attention(self, inputs, output_size, gene, variation, activation_fn=tf.tanh):
        inputs_shape = inputs.get_shape()
        if len(inputs_shape) != 3 and len(inputs_shape) != 4:
            raise ValueError('Shape of input must have 3 or 4 dimensions')
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn)
        doc_context = tf.concat([gene, variation], axis=1)
        doc_context_vector = layers.fully_connected(doc_context, output_size,
                                                    activation_fn=activation_fn)
        doc_context_vector = tf.expand_dims(doc_context_vector, 1)
        if len(inputs_shape) == 4:
            doc_context_vector = tf.expand_dims(doc_context_vector, 1)

        vector_attn = input_projection * doc_context_vector
        vector_attn = tf.reduce_sum(vector_attn, axis=-1, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = input_projection * attention_weights
        outputs = tf.reduce_sum(weighted_projection, axis=-2)

        return outputs


if __name__ == '__main__':
    main(ModelHAN(), 'han', sentence_split=True, batch_size=TC_BATCH_SIZE_HATT)
