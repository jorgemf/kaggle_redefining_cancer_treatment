import tensorflow as tf
import numpy as np
from itertools import groupby
from src.configuration import *
from src.tf_dataset import TFDataSet


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
        elif type == 'test':
            data_files = os.path.join(DIR_DATA_TEXT_CLASSIFICATION, 'test_set')
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
                                                        shuffle_size=10000,
                                                        padded_shapes=[],
                                                        padded_values=-1.0)

    def _map(self, example_serialized):
        example_serialized = example_serialized.split()
        sequence = [np.int32(w) for w in example_serialized[1:]]
        if self.sentence_split is not None:
            groups = groupby(sequence, lambda x: x == self.sentence_split)
            sequence = [list(g) for k, g in groups if not k]
            if len(sequence) > MAX_SENTENCES:
                sequence = sequence[:MAX_SENTENCES]
            while len(sequence) < MAX_SENTENCES:
                sequence.append([-1] * MAX_WORDS_IN_SENTENCE)
            for i, sentence in enumerate(sequence):
                if len(sentence) > MAX_WORDS_IN_SENTENCE:
                    sentence = sentence[:MAX_WORDS_IN_SENTENCE]
                sequence[i] = np.asarray(sentence, dtype=np.int32)
        else:
            if len(sequence) > MAX_WORDS:
                sequence = sequence[:MAX_WORDS]

        if self.type == 'train':
            # first class is 1, last one is 9
            data_sample_class = int(example_serialized[0]) - 1
            return [
                np.asarray(sequence, dtype=np.int32),
                np.asarray([data_sample_class], dtype=np.int32)
            ]
        elif self.type == 'test':
            return np.asarray(sequence, dtype=np.int32)
        else:
            raise ValueError()
