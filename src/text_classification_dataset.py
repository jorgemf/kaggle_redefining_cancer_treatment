import tensorflow as tf
import numpy as np
from itertools import groupby
from configuration import *
from dataset_filelines import DatasetFilelines


class TextClassificationDataset(DatasetFilelines):
    """
    Helper class for the dataset. See dataset_filelines.DatasetFilelines for more details.
    """

    def __init__(self, type='train', sentence_split=False):
        """
        :param str type: type of set, either 'train' or 'test'
        """
        if type == 'train':
            data_files = [os.path.join(DIR_DATA_TEXT_CLASSIFICATION, 'train_set')]
        elif type == 'test':
            data_files = [os.path.join(DIR_DATA_TEXT_CLASSIFICATION, 'test_set')]
        else:
            raise ValueError('Type can only be train or test but it is {}'.format(type))
        self.type = type
        self.sentence_split = None
        if sentence_split:
            dict_filename = 'word2vec_{}_{}'.format(VOCABULARY_SIZE, EMBEDDINGS_SIZE)
            dict_filepath = os.path.join(DIR_DATA_WORD2VEC, dict_filename)
            with tf.gfile.FastGFile(dict_filepath, 'r') as f:
                for line in f:
                    data = line.split()
                    symbol = data[0]
                    if symbol == '.':
                        self.sentence_split = int(data[1])
                        break

        super(TextClassificationDataset, self).__init__(name=type, data_files=data_files,
                                                        min_queue_examples=TC_BATCH_SIZE)

    def py_func_parse_example(self, example_serialized):
        example_serialized = example_serialized.split()
        data_sample_class = -1
        try:
            data_sample_class = int(example_serialized[0]) - 1  # first class is 1, last one is 9
        except:
            pass
        sequence = [np.int32(w) for w in example_serialized[1:]]
        if self.sentence_split is not None:
            groups = groupby(sequence, lambda x: x == self.sentence_split)
            sequence = [list(g) for k, g in groups if not k]
            if len(sequence) > MAX_SENTENCES:
                sequence = sequence[:MAX_SENTENCES]
            while len(sequence) < MAX_SENTENCES:
                sequence.append([-1] * MAX_WORDS_IN_SENTENCE)
            for i, sentence in enumerate(sequence):
                if len(sentence) > MAX_SENTENCES:
                    sentence = sentence[:MAX_WORDS_IN_SENTENCE]
                while len(sentence) < MAX_WORDS_IN_SENTENCE:
                    sentence.append(-1)
                sequence[i] = sentence
        else:
            if len(sequence) > MAX_WORDS:
                sequence = sequence[:MAX_WORDS]
            # add padding
            while len(sequence) < MAX_WORDS:
                sequence.append(-1)
        return [
            np.asarray(sequence, dtype=np.int32),
            np.asarray([data_sample_class], dtype=np.int32)
        ]

    def py_func_parse_example_types(self):
        return [tf.int32, tf.int32]

    def py_func_parse_example_inputs_outputs(self):
        return 1, 1

    def py_fun_parse_example_reshape(self, inputs, outputs):
        if self.sentence_split is not None:
            inputs[0] = tf.reshape(inputs[0], [MAX_SENTENCES])
        else:
            inputs[0] = tf.reshape(inputs[0], [MAX_WORDS])
        outputs[0] = tf.reshape(outputs[0], [1])
        return inputs, outputs
