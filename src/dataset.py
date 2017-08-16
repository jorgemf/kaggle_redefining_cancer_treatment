import multiprocessing
import logging
import tensorflow as tf


class Dataset(object):
    def __init__(self, name, min_queue_examples, num_preprocess_threads=None, num_readers=None):
        """
        :param name: name of the dataset.
        :param int min_queue_examples: minimum number of examples to queue, this value should be
        proportional to the ram of the computer. Bigger values provide a better shuffling
        of the data
        :param int num_preprocess_threads: Number of threads to preprocess the data. By default is
        the number of CPUs.
        :param int num_readers: Number of readers to read the data. By default is the minimum value
        between the number of CPUs divided by 2 and the number of data files
        """
        self.name = name
        self.min_queue_examples = min_queue_examples
        if num_preprocess_threads is None:
            self.num_preprocess_threads = multiprocessing.cpu_count()
        else:
            self.num_preprocess_threads = num_preprocess_threads
        if num_readers is None:
            self.num_readers = min(int(self.num_preprocess_threads / 2),
                                   len(self.data_files_names()))
            if self.num_readers < 1:
                self.num_readers = 1
        else:
            self.num_readers = num_readers
        if self.min_queue_examples < 1:
            raise ValueError('Please make min_queue_examples at least 1')
        if self.num_readers < 1:
            raise ValueError('Please make num_readers at least 1')
        if self.num_preprocess_threads < 1:
            raise ValueError('Please make num_preprocess_threads at least 1')

    def read(self, batch_size, shuffle=False):
        """
        Reads the data and return a tuple of (inputs,outputs)
        :param batch_size: the bactch size of the returned inputs/outputs
        :param shuffle: whether to shuffle the data or not
        :return: a tuple of (inputs,outputs) with the batch size set
        """
        data_file_names = self.data_files_names()

        filename_queue = tf.train.string_input_producer(data_file_names, shuffle=shuffle)

        capacity_examples_queue = self.min_queue_examples + \
                                  (self.num_preprocess_threads + 2) * batch_size
        if shuffle:
            examples_queue = tf.RandomShuffleQueue(
                capacity=capacity_examples_queue,
                min_after_dequeue=self.min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(capacity=capacity_examples_queue, dtypes=[tf.string])

        if self.num_readers > len(data_file_names):
            logging.warn('you have more readers than data files, this could lead to race '
                         'conditions when reading the data')
        if self.num_readers > 1:
            enqueue_ops = []
            for _ in range(self.num_readers):
                reader = self.get_reader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = self.get_reader()
            _, example_serialized = reader.read(filename_queue)
        inputs_outputs = []
        for thread_id in range(self.num_preprocess_threads):
            inputs, outputs = self.parse_example(example_serialized)
            data = []
            if isinstance(inputs, list):
                input_list_size = len(inputs)
                data.extend(inputs)
            elif inputs is not None:
                input_list_size = 1
                data.append(inputs)
            else:
                input_list_size = 0
            if isinstance(outputs, list):
                output_list_size = len(outputs)
                data.extend(outputs)
            elif outputs is not None:
                output_list_size = 1
                data.append(outputs)
            else:
                output_list_size = 0
            inputs_outputs.append(data)

        capacity = 2 * self.num_preprocess_threads * batch_size
        if shuffle:
            result = tf.train.shuffle_batch_join(inputs_outputs, batch_size=batch_size,
                                                 min_after_dequeue=capacity / 2, capacity=capacity)
        else:
            result = tf.train.batch_join(inputs_outputs, batch_size=batch_size, capacity=capacity)
        if input_list_size == 0:
            inputs = None
            outputs = result
        elif input_list_size == 1:
            inputs = result[0]
        elif output_list_size != 0:
            inputs = result[:input_list_size]
        if output_list_size == 0:
            outputs = None
            inputs = result
        elif output_list_size == 1:
            outputs = result[-1]
        elif input_list_size != 0:
            outputs = result[-output_list_size:]
        inputs = self.preprocess_inputs(inputs)
        outputs = self.preprocess_outputs(outputs)

        return inputs, outputs

    def get_reader(self):
        """
        :return: a reader for a single entry from the data set.
        """
        raise NotImplementedError('Should have implemented this')

    def data_files_names(self):
        """
        :return: python list of all (sharded) dataset files.
        """
        raise NotImplementedError('Should have implemented this')

    def get_size(self):
        """
        :return: The number of items of the dataset
        """
        raise NotImplementedError('Should have implemented this')

    def parse_example(self, example_serialized):
        """
        Process a serialized example from the data files and returns a tuple with the inputs and
        outputs of the example
        :return: a tuple (inputs,outputs) with the input data for the model and the expected output
        """
        raise NotImplementedError('Should have implemented this')

    def preprocess_inputs(self, inputs):
        """
        Override this function to process the inputs when they are read.
        :param inputs: The inputs
        :return: the modified inputs
        """
        return inputs

    def preprocess_outputs(self, outputs):
        """
        Override this function to process the outputs when they are read.
        :param outputs: The outputs
        :return: the modified outputs
        """
        return outputs
