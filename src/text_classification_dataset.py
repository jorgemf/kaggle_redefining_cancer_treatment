import tensorflow as tf
import numpy as np
from itertools import groupby
from .configuration import *
from .tf_dataset import TFDataSet


class TextClassificationDataset(TFDataSet):
    """
    Helper class for the dataset. See dataset_filelines.DatasetFilelines for more details.
    """

    def __init__(self, type='train', sentence_split=False):
        """
        :param str type: type of set, either 'train' or 'test'
        """
        if type == 'train':
            data_files = os.path.join(DIR_DATA_TEXT_CLASSIFICATION, 'train_set')
            if sentence_split:
                padded_shape = ([None, MAX_WORDS_IN_SENTENCE], [1])
            else:
                padded_shape = ([None], [1])
            padded_values = (-1, -1)
        elif type == 'test':
            data_files = os.path.join(DIR_DATA_TEXT_CLASSIFICATION, 'test_set')
            if sentence_split:
                padded_shape = [None, MAX_WORDS_IN_SENTENCE]
            else:
                padded_shape = [None]
            padded_values = -1
        else:
            raise ValueError('Type can only be train or test but it is {}'.format(type))
        self.type = type
        self.sentence_split = None
        if sentence_split:
            dict_filename = 'word2vec_dataset_{}_dict'.format(VOCABULARY_SIZE)
            dict_filepath = os.path.join(DIR_DATA_WORD2VEC, dict_filename)
            with tf.gfile.FastGFile(dict_filepath, 'r') as f:
                for line in f:
                    data = line.split()
                    symbol = data[0]
                    if symbol == '.':
                        self.sentence_split = int(data[1])
                        break

        super(TextClassificationDataset, self).__init__(name=type,
                                                        data_files_pattern=data_files,
                                                        min_queue_examples=100,
                                                        shuffle_size=10000)
        # TODO TF <= 1.2.0 have an issue with padding with more than one dimension
        #                                                 padded_shapes=padded_shape,
        #                                                 padded_values=padded_values)

    def _map(self, example_serialized):
        def _parse_sequence(example_serialized, dataset_type):
            example_serialized = example_serialized.split()
            sequence = list([np.int32(w) for w in example_serialized[1:]])
            if self.sentence_split is not None:
                groups = groupby(sequence, lambda x: x == self.sentence_split)
                sequence = list([list(g) for k, g in groups if not k])
                sentence_padding = [-1] * MAX_WORDS_IN_SENTENCE
                for i, sentence in enumerate(sequence):
                    if len(sentence) > MAX_WORDS_IN_SENTENCE:
                        sentence = sentence[:MAX_WORDS_IN_SENTENCE]
                    else:
                        # TODO added padding for TF <= 1.2.0
                        sentence.extend([-1] * (MAX_WORDS_IN_SENTENCE - len(sentence)))
                    sequence[i] = np.asarray(sentence, dtype=np.int32)
                if len(sequence) > MAX_SENTENCES:
                    sequence = sequence[:MAX_SENTENCES]
                else:
                    # TODO added padding for TF <= 1.2.0
                    sequence.extend([sentence_padding] * (MAX_SENTENCES - len(sequence)))
            else:
                if len(sequence) > MAX_WORDS:
                    sequence = sequence[:MAX_WORDS]
                else:
                    # TODO added padding for TF <= 1.2.0
                    sequence.extend([-1] * (MAX_WORDS - len(sequence)))

            if dataset_type == 'train':
                # first class is 1, last one is 9
                data_sample_class = int(example_serialized[0]) - 1
                return [
                    np.asarray(sequence, dtype=np.int32),
                    np.asarray([data_sample_class], dtype=np.int32)
                ]
            elif dataset_type == 'test':
                return np.asarray(sequence, dtype=np.int32)
            else:
                raise ValueError()


        if self.type == 'train':
            sequence, result_class = tf.py_func(lambda x: _parse_sequence(x, self.type),
                                                [example_serialized], [tf.int32, tf.int32],
                                                stateful=True)
            # TODO for TF <= 1.2.0  set shape because of padding
            if self.sentence_split is not None:
                sequence = tf.reshape(sequence,[MAX_SENTENCES, MAX_WORDS_IN_SENTENCE])
            else:
                sequence = tf.reshape(sequence,[MAX_WORDS])

            sequence = tf.reshape(sequence, [-1])
            result_class = tf.reshape(result_class, [1])
            return sequence, result_class
        elif type == 'test':
            sequence = tf.py_func(lambda x: _parse_sequence(x, self.type),
                                  [example_serialized], [tf.int32], stateful=True)
            # TODO for TF <= 1.2.0  set shape because of padding
            if self.sentence_split is not None:
                sequence = tf.reshape(sequence,[MAX_SENTENCES, MAX_WORDS_IN_SENTENCE])
            else:
                sequence = tf.reshape(sequence,[MAX_WORDS])
            return sequence
        else:
            raise ValueError()
