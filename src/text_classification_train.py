import tensorflow as tf
import math
import numpy as np
import random
import csv
import time
from datetime import timedelta
from tensorport_template import trainer
from tensorflow.python.training import training_util
from configuration import *
import text_classification_model_simple as text_classification_model
from dataset_filelines import DatasetFilelines


class TextClassificationDataset(DatasetFilelines):
    """
    Helper class for the dataset. See dataset_filelines.DatasetFilelines for more details.
    """

    def __init__(self, type='train'):
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
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
        # add padding
        while len(sequence) < MAX_SEQUENCE_LENGTH:
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
        inputs[0] = tf.reshape(inputs[0], [MAX_SEQUENCE_LENGTH])
        outputs[0] = tf.reshape(outputs[0], [1])
        return inputs, outputs


class TextClassificationTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, epochs=TC_EPOCHS):
        self.dataset = dataset
        max_steps = epochs * dataset.get_size()
        super(TextClassificationTrainer, self).__init__(DIR_TC_LOGDIR, max_steps=max_steps)

    def _load_embeddings(self, vocabulary_size, embeddings_size):
        embeddings = []
        embeddings_file = 'embeddings_{}_{}'.format(vocabulary_size, embeddings_size)
        with open(os.path.join(DIR_DATA_WORD2VEC, embeddings_file), 'r') as file:
            reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                embeddings.append([float(r) for r in row])
        return embeddings

    def model(self,
              batch_size=TC_BATCH_SIZE,
              vocabulary_size=VOCABULARY_SIZE,
              embeddings_size=EMBEDDINGS_SIZE,
              output_classes=9,
              learning_rate_initial=TC_LEARNING_RATE_INITIAL,
              learning_rate_decay=TC_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=TC_LEARNING_RATE_DECAY_STEPS):
        # embeddings
        embeddings = self._load_embeddings(vocabulary_size, embeddings_size)

        # global step
        self.global_step = training_util.get_or_create_global_step()

        # inputs
        self.inputs_text, self.expected_labels = self.dataset.read(batch_size, shuffle=True)

        # model
        outputs = text_classification_model.model(self.inputs_text, output_classes,
                                                  embeddings=embeddings)

        # loss
        targets = text_classification_model.targets(self.expected_labels, output_classes)
        self.loss = text_classification_model.loss(targets, outputs)
        tf.summary.scalar('loss', self.loss)
        # measure other errors
        targets = tf.squeeze(targets)
        mse_mean = tf.losses.mean_squared_error(targets, outputs['prediction_mean'],
                                                loss_collection=None)
        mse_percent = tf.losses.mean_squared_error(targets, outputs['prediction_percent'],
                                                   loss_collection=None)
        tf.summary.scalar('mse_prediction_mean', mse_mean)
        tf.summary.scalar('mse_prediction_percent', mse_percent)

        # learning rate
        self.learning_rate = tf.train.exponential_decay(learning_rate_initial, self.global_step,
                                                        learning_rate_decay_steps,
                                                        learning_rate_decay,
                                                        staircase=True, name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)

        # optimizer and gradient clipping
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables),
                                                   global_step=self.global_step)

        # summaries and moving average
        ema = tf.train.ExponentialMovingAverage(0.9)
        ema_assign = ema.apply([self.loss])
        with tf.control_dependencies([self.optimizer]):
            self.training_op = tf.group(ema_assign)
        self.average_loss = ema.average(self.loss)
        # saver to save the model
        self.saver = tf.train.Saver()
        # check a nan value in the loss
        self.loss = tf.check_numerics(self.loss, 'loss is nan')

        return None

    def create_graph(self):
        return self.model()

    def train_step(self, session, graph_data):
        lr, _, _, loss_val, step = session.run([self.learning_rate, self.training_op,
                                                self.loss, self.average_loss,
                                                self.global_step])
        # if self.is_chief and step % 10 == 0:
        if self.is_chief:
            elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
            m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
            print(m.format(step, loss_val, lr, elapsed_time))
        pass

    def after_create_session(self, session, coord):
        self.init_time = time.time()


if __name__ == '__main__':
    # start the training
    TextClassificationTrainer(dataset=TextClassificationDataset()).train()
