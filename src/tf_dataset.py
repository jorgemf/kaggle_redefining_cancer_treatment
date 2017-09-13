import multiprocessing
import tensorflow as tf
from tensorflow.contrib.data import TextLineDataset
from tensorflow.contrib.data import Dataset
from tensorflow.python.framework.errors_impl import OutOfRangeError


class TFDataSet(object):
    """Abstract class that helps to work with TensorFlow Datasets"""

    def __init__(self, name, data_files_pattern, dataset_class=TextLineDataset,
                 min_queue_examples=0, shuffle_size=None, padded_shapes=None, padded_values=None):
        """
        :param name: name of the dataset.
        :param str data_files_pattern: pattern of the data files
        :param Dataset dataset_class: class to create the dataset with the files. By default is
        TextLineDataset
        :param int min_queue_examples: minimum number of examples to queue, this value should be
        proportional to the ram of the computer. By default is 0
        :param int shuffle_size: size of the buffer for shuffling, this value should be
        proportional to the ram of the computer
        :param List[tf.Tensor] padded_shapes: shape for padding the batch
        :param tf.Tensor padded_values: values for the padding
        """
        self.name = name
        self.data_files_pattern = data_files_pattern
        self.dataset_class = dataset_class
        self.min_queue_examples = min_queue_examples
        self.shuffle_size = shuffle_size
        self.padded_shapes = padded_shapes
        self.padded_values = padded_values

    def read(self, batch_size, num_epochs=1, shuffle=False, task_spec=None):
        """
        Reads the data and return a tuple of (inputs,outputs)
        :param batch_size: the batch size of the returned inputs/outputs
        :param num_epochs: the number of epochs to read the dataset
        :param shuffle: whether to shuffle the data or not
        :param task_spec: the task spec of the training. I will help to know whether it is
        distributed training or not
        :return: The result of calling dataset.make_one_shot_iterator().get_next()
        """
        # create the dataset of files with the data
        # TODO in TF 1.3 use:  dataset = Dataset.list_files(self.data_files_pattern)
        from tensorflow.python.ops import gen_io_ops
        dataset = Dataset.from_tensor_slices(gen_io_ops.matching_files(self.data_files_pattern))
        # set the number of epochs
        dataset = dataset.repeat(num_epochs)
        if shuffle:
            # read one sample per file
            # TODO in TF 1.3 use:
            # dataset = dataset.interleave(self.dataset_class,
            #                              # number of readers the same as number of CPUs
            #                              cycle_length=multiprocessing.cpu_count() + 1,
            #                              # block size is 1 to get directly a flat map
            #                              block_length=1)
            files = []
            filename = dataset.make_one_shot_iterator().get_next()
            try:
                with tf.Session() as sess:
                    while True:
                        d = sess.run(filename)
                        files.append(d)
            except OutOfRangeError:
                pass
            dataset = self.dataset_class(files)
        else:
            # reads files sequentially
            files = []
            filename = dataset.make_one_shot_iterator().get_next()
            try:
                with tf.Session() as sess:
                    while True:
                        d = sess.run(filename)
                        files.append(d)
            except OutOfRangeError:
                pass
            dataset = self.dataset_class(files)

        if task_spec and task_spec.num_workers > 1:
            # split the dataset in shards
            # TODO in TF 1.4 use: dataset = dataset.shard(task_spec.num_workers, task_spec.index)
            from tensorflow.python.ops import math_ops

            def filter_fn(elem_index, _):
                mod_result = math_ops.mod(elem_index, task_spec.num_workers)
                return math_ops.equal(mod_result, task_spec.index)

            dataset = dataset.enumerate().filter(filter_fn).map(lambda _, elem: elem)

        if shuffle:
            # shuffle the samples
            if self.shuffle_size is None:
                raise ValueError('shuffle_size has not been set')
            dataset = dataset.shuffle(buffer_size=self.shuffle_size)

        # process each example. We check the method is defined in the child class:
        if self._flat_map.__func__ not in TFDataSet.__dict__.values():
            dataset = dataset.flat_map(self._flat_map)
        if self._map.__func__ not in TFDataSet.__dict__.values():
            dataset = dataset.map(self._map,
                                  # use as many threads as CPUs + 1
                                  # TODO in TF 1.4 use: num_parallel_calls=multiprocessing.cpu_count() + 1,
                                  num_threads=multiprocessing.cpu_count() + 1,
                                  # buffer the data as CPUs * batch_size + minimum_size
                                  output_buffer_size=batch_size * multiprocessing.cpu_count() +
                                                     self.min_queue_examples)
        if self.padded_shapes:
            dataset = dataset.padded_batch(batch_size, self.padded_shapes, self.padded_values)
        else:
            dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    # TODO remove features in TF 1.3
    def _flat_map(self, example_serialized, features=None):
        """
        Flat maps the example serialized.
        Simple example:

        def _parse(line):
            a, b = [np.int32(x) for x in line.split()]
            return [a, a*2, a*3], [b, b*2, b*3]

        t_input, t_ouptut = tf.py_func(_parse, [line], [tf.int32, tf.int32],
                                       stateful=True, name='py_parse_example')
        v = (t_intput, t_output)
        return Dataset.from_tensor_slices(v)

        :param example_serialized:
        :param features: do not use this as it is deprecated after 1.2
        :return: a dataset
        """
        pass

    # TODO remove features in TF 1.3
    def _map(self, example_serialized, features=None):
        """
        Maps a example_serialized read from the dataset into the final set of tf.Tensors
        to return to the model.

        Simple example:

        def _parse(line, features=None):
            a, b = [np.int32(x) for x in line.split()]
            return a, b

        t_input, t_ouptut = tf.py_func(_parse, [line], [tf.int32, tf.int32],
                                       stateful=True, name='py_parse_example')
        t_ouptut = tf.add(t_ouptut, 1)

        return t_input, t_ouptut

        :param example_serialized: the example serialized
        :param features: do not use this as it is deprecated after 1.2
        :return: a tuple of the tensors to return when get_next is called. Usually (inputs,outputs)
        """
        pass

    def _count_num_records(self):
        """
        Counts the number of non-empty lines (the data samples) from the data_files. This function
        is called from get_size the first time.
        :return int: the number of non-empty lines in the data_files
        """
        size = 0
        # TODO in TF 1.3 use: dataset = Dataset.list_files(self.data_files_pattern).repeat(1)
        from tensorflow.python.ops import gen_io_ops
        dataset = Dataset.from_tensor_slices(
            gen_io_ops.matching_files(self.data_files_pattern)).repeat(1)

        dataset = self.dataset_class(dataset).repeat(1)
        samples = 0
        try:
            next_element = dataset.make_one_shot_iterator().get_next()
            with tf.Session() as sess:
                while True:
                    sess.run(next_element)
                    samples += 1
        except:
            pass
        return samples

    def get_size(self):
        if self._size is None:
            self._size = self._count_num_records()
        return self._size
