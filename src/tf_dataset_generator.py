import multiprocessing
import tensorflow as tf
from tensorflow.contrib.data import Dataset

# from TensorFlow 1.4
import collections
import threading
from tensorflow.python.ops import script_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape


# from TensorFlow 1.4
class _GeneratorState(object):
    def __init__(self, generator):
        self._generator = generator
        self._lock = threading.Lock()
        self._next_id = 0
        self._iterators = collections.defaultdict(lambda: iter(generator()))

    def get_next_id(self):
        with self._lock:
            ret = self._next_id
            self._next_id += 1
        return ret

    def get_iterator(self, iterator_id):
        return self._iterators[iterator_id]

    def iterator_completed(self, iterator_id):
        del self._iterators[iterator_id]


class TFDataSetGenerator(object):
    """Abstract class that helps to work with TensorFlow Datasets"""

    def __init__(self, name, generator, output_types, output_shapes=None, min_queue_examples=0,
                 shuffle_size=None, padded_shapes=None, padded_values=None):
        """
        :param name: name of the dataset.
        :param generator generator: generator of elements in of the dataset
        :param output_types: list of output types of the generator
        :param output_shapes: output shapes of the generator
        :param int min_queue_examples: minimum number of examples to queue, this value should be
        proportional to the ram of the computer. By default is 0
        :param int shuffle_size: size of the buffer for shuffling, this value should be
        proportional to the ram of the computer
        :param List[tf.Tensor] padded_shapes: shape for padding the batch
        :param tf.Tensor padded_values: values for the padding
        """
        if not callable(generator):
            raise TypeError("`generator` must be callable.")
        self.name = name
        self.generator = generator
        self.output_types = output_types
        self.output_shapes = output_shapes
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
        # TODO in TF 1.4 use: dataset = Dataset.from_generator(self.generator)
        # FIXME repeat doesn't work with generators, so we can encapsulate the generator here
        def _epochs():
            for _ in range(num_epochs):
                for item in self.generator():
                    yield item
        generator_state = _GeneratorState(_epochs)
        output_types = self.output_types
        output_shapes = self.output_shapes
        output_shapes = nest.map_structure(
            lambda _: tensor_shape.TensorShape(None), output_types)
        flattened_types = nest.flatten(output_types)
        flattened_shapes = nest.flatten(output_shapes)

        def get_iterator_id_map_fn(dummy):
            return script_ops.py_func(generator_state.get_next_id, [], tf.int64, stateful=True)

        def generator_map_fn(iterator_id_t):
            def generator_py_func(iterator_id):
                try:
                    values = next(generator_state.get_iterator(iterator_id))
                except StopIteration:
                    generator_state.iterator_completed(iterator_id)
                    raise StopIteration("Iteration finished.")
                ret_arrays = [
                    script_ops.FuncRegistry._convert(ret)
                    for ret in nest.flatten_up_to(output_types, values)
                ]
                for (ret_array, expected_dtype, expected_shape) in zip(
                        ret_arrays, flattened_types, flattened_shapes):
                    if ret_array.dtype != expected_dtype.as_numpy_dtype:
                        raise TypeError(
                            "`generator` yielded an element of type %s where an element "
                            "of type %s was expected." % (ret_array.dtype,
                                                          expected_dtype.as_numpy_dtype))
                    if not expected_shape.is_compatible_with(ret_array.shape):
                        raise ValueError(
                            "`generator` yielded an element of shape %s where an element "
                            "of shape %s was expected." % (ret_array.shape, expected_shape))
                return ret_arrays

            flat_values = script_ops.py_func(generator_py_func, [iterator_id_t], self.output_types,
                                             stateful=True)
            if output_shapes is not None:
                for ret_t, shape in zip(flat_values, flattened_shapes):
                    ret_t.set_shape(shape)
            return nest.pack_sequence_as(output_types, flat_values)

        def flat_map_fn(iterator_id_t):
            repeated_id = Dataset.from_tensors(iterator_id_t).repeat(None)
            return repeated_id.map(generator_map_fn)

        id_dataset = Dataset.from_tensors(0).map(get_iterator_id_map_fn)
        dataset = id_dataset.flat_map(flat_map_fn)
        ############################################################################################

        # set the number of epochs
        # FIXME repeat doesn't work with generators
        # dataset = dataset.repeat(num_epochs)

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
        if self._map.__func__ not in TFDataSetGenerator.__dict__.values():
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
    def _map(self, example, features=None):
        """
        See TFDataSet._map()
        """
        pass
