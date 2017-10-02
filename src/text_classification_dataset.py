import tensorflow as tf
import numpy as np
from itertools import groupby
from .configuration import *
from .tf_dataset import TFDataSet


def _padding(arr, pad, token=-1):
    len_arr = len(arr)
    if len_arr > pad:
        return arr[:pad]
    elif len_arr < pad:
        arr.extend([token] * (pad - len_arr))
        return arr
    else:
        return arr


class TextClassificationDataset(TFDataSet):
    """
    Helper class for the dataset. See dataset_filelines.DatasetFilelines for more details.
    """

    def __init__(self, type='train', sentence_split=False):
        """
        :param str type: type of set, either 'train' or 'test'
        """
        data_files = os.path.join(DIR_DATA_TEXT_CLASSIFICATION, '{}_set'.format(type))
        if type == 'train' or type == 'val':
            if sentence_split:
                padded_shape = ([None, MAX_WORDS_IN_SENTENCE], [1])
            else:
                padded_shape = ([None], [1])
            padded_values = (-1, -1)
        elif type == 'test' or type == 'stage2_test':
            if sentence_split:
                padded_shape = [None, MAX_WORDS_IN_SENTENCE]
            else:
                padded_shape = [None]
            padded_values = -1
        else:
            raise ValueError(
                    'Type can only be train, val, test or stage2_test but it is {}'.format(type))
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
        variant_padding = 20
        sentence_padding = [-1] * MAX_WORDS_IN_SENTENCE

        def _parse_sequence(example_serialized, dataset_type):
            example_serialized = example_serialized.split('||')
            example_class = example_serialized[0].strip()
            example_gene = np.int32(example_serialized[1].strip())
            example_variant = list([np.int32(w) for w in example_serialized[2].strip().split()])
            sequence = list([np.int32(w) for w in example_serialized[3].strip().split()])

            example_variant = _padding(example_variant, variant_padding)

            if self.sentence_split is not None:
                groups = groupby(sequence, lambda x: x == self.sentence_split)
                sequence = list([list(g) for k, g in groups if not k])
                for i, sentence in enumerate(sequence):
                    sentence = _padding(sentence, MAX_WORDS_IN_SENTENCE)
                    sequence[i] = np.asarray(sentence, dtype=np.int32)
                sequence_begin = _padding(sequence, MAX_SENTENCES, token=sentence_padding)
                sequence_end = _padding(list(reversed(sequence)), MAX_SENTENCES, token=sentence_padding)
            else:
                sequence_begin = _padding(sequence, MAX_WORDS)
                sequence_end = _padding(list(reversed(sequence)), MAX_WORDS, token=sentence_padding)

            if dataset_type == 'train' or dataset_type == 'val':
                # first class is 1, last one is 9
                data_sample_class = int(example_class) - 1
                return [
                    np.asarray(sequence_begin, dtype=np.int32),
                    np.asarray(sequence_end, dtype=np.int32),
                    np.int32(example_gene),
                    np.asarray(example_variant, dtype=np.int32),
                    np.int32(data_sample_class),
                    ]
            elif dataset_type == 'test' or dataset_type == 'stage2_test':
                return [
                    np.asarray(sequence_begin, dtype=np.int32),
                    np.asarray(sequence_end, dtype=np.int32),
                    np.int32(example_gene),
                    np.asarray(example_variant, dtype=np.int32),
                    ]
            else:
                raise ValueError()

        if self.type == 'train' or self.type == 'val':
            sequence_begin, sequence_end, gene, variant, result_class = \
                tf.py_func(lambda x: _parse_sequence(x, self.type), [example_serialized],
                           [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32], stateful=True)
        elif self.type == 'test' or self.type == 'stage2_test':
            sequence_begin, sequence_end, gene, variant = \
                tf.py_func(lambda x: _parse_sequence(x, self.type), [example_serialized],
                           [tf.int32, tf.int32, tf.int32, tf.int32], stateful=True)
            result_class = None
        else:
            raise ValueError()

        # TODO for TF <= 1.2.0  set shape because of padding
        if self.sentence_split is not None:
            sequence_begin = tf.reshape(sequence_begin, [MAX_SENTENCES, MAX_WORDS_IN_SENTENCE])
        else:
            sequence_end = tf.reshape(sequence_end, [MAX_WORDS])

        gene = tf.reshape(gene, [1])
        variant = tf.reshape(variant, [variant_padding])
        if result_class is not None:
            result_class = tf.reshape(result_class, [1])
            return sequence_begin, sequence_end, gene, variant, result_class
        else:
            return sequence_begin, sequence_end, gene, variant

