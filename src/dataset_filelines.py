import tensorflow as tf
from tensorflow.python.platform import gfile
from dataset import Dataset


class DatasetFilelines(Dataset):
    def __init__(self, name, data_files, min_queue_examples, num_preprocess_threads=None,
                 num_readers=None):
        """
        :param name: see Dataset
        :param data_dir: directory with the tf records with the format name-*
        :param min_queue_examples: see Dataset
        :param num_preprocess_threads: see Dataset
        :param num_readers: see Dataset
        """
        self._size = None
        self.data_files = data_files
        super(DatasetFilelines, self).__init__(name=name,
                                               min_queue_examples=min_queue_examples,
                                               num_preprocess_threads=num_preprocess_threads,
                                               num_readers=num_readers)

    def _count_num_records(self):
        """
        Counts the number of non-empty lines (the data samples) from the data_files. This function
        is called from get_size the first time.
        :return int: the number of non-empty lines in the data_files
        """
        size = 0
        for data_file in self.data_files:
            with gfile.FastGFile(data_file, 'r') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        size += 1
        return size

    def get_size(self):
        if self._size is None:
            self._size = self._count_num_records()
        return self._size

    def parse_example(self, example_serialized):
        inputs_num, outputs_num = self.py_func_parse_example_inputs_outputs()
        if inputs_num < 0:
            raise ValueError('Number of inputs is <0')
        if outputs_num < 0:
            raise ValueError('Number of outputs is <0')
        output_types = self.py_func_parse_example_types()
        if len(output_types) != inputs_num + outputs_num:
            raise ValueError('The number of inputs ({}) and outputs ({}) does not match the '
                             'number of types ({}) for the py_func_parse_example function'
                             .format(inputs_num, outputs_num, len(output_types)))
        for t in output_types:
            if t not in [tf.int8, tf.int16, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64,
                         tf.string, tf.bool, tf.uint8, tf.uint16, tf.complex64, tf.complex128,
                         tf.qint8, tf.qint16, tf.qint32, tf.quint8, tf.quint16]:
                raise ValueError('type "{}" not recognized as a valid tf type'.format(t))

        with tf.device('/cpu'):
            parsed_example = tf.py_func(self.py_func_parse_example, [example_serialized],
                                        output_types, stateful=True, name='parse_example')
        if inputs_num > 0:
            inputs = parsed_example[:inputs_num]
        else:
            inputs = None
        if outputs_num > 0:
            outputs = parsed_example[-outputs_num:]
        else:
            outputs = None
        inputs, outpts = self.py_fun_parse_example_reshape(inputs, outputs)
        return inputs, outputs

    def py_func_parse_example(self, example_serialized):
        """
        Given a example serialized this function returns a list of tensors with the example
        deserialized.
        A Python function, which accepts a list of NumPy ndarray objects having element types that
        match the corresponding tf.Tensor objects in inp, and returns a list of ndarray objects
        (or a single ndarray) having element types that match the corresponding values in Tout.
        :param example_serialized:
        :return List[tf.Tensor]: List of tensors with the
        """
        raise NotImplementedError('Should have implemented this')

    def py_func_parse_example_types(self):
        """
        :return List[]: a list of tf.types that the py_func_parse_example returns.
        """
        raise NotImplementedError('Should have implemented this')

    def py_func_parse_example_inputs_outputs(self):
        """
        :return (int,int): a tuple with two integers: the number of inputs and the number of outputs
        that the function py_func_parse_example will return. By default returns (1,1)
        """
        return 1, 1

    def py_fun_parse_example_reshape(self, inputs, outputs):
        """
        The data processed with py_func_parse_example needs to reshape to give the tensors shape,
        otherwise the batch wont work.
        :param inputs: list of inputs tensors
        :param outputs: list of output tensors
        :return: the inputs and outputs reshaped
        """
        raise NotImplementedError('Should have implemented this')

    def get_reader(self):
        return tf.TextLineReader()

    def data_files_names(self):
        return self.data_files
