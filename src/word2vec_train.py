import tensorflow as tf
import math
import numpy as np
import random
import csv
import time
from datetime import timedelta
import shutil
from tensorflow.python.training import training_util
from tensorflow.contrib.tensorboard.plugins import projector
from src import trainer
from src.word2vec_process_data import load_word2vec_data
from src.configuration import *
from dataset import Dataset


def select_random_labels(labels, num_labels, probabilities):
    """
    Selects random labels from a labels list
    :param List[int] labels: list of labels
    :param int num_labels: number of labels to select
    :param List[float] probabilities: weighted probabilities to select a label
    :return List[int]: list of selected labels
    """
    if len(labels) <= num_labels:
        return labels
    samples = []
    probabilities_copy = list(probabilities)
    probabilities_sum = np.sum(probabilities_copy)
    for _ in range(num_labels):
        r = random.random() * probabilities_sum
        p_sum = 0.0
        i = 0
        while p_sum < r:
            p_sum += probabilities_copy[i]
            i += 1
        i -= 1
        samples.append(labels[i])
        probabilities_sum -= probabilities_copy[i]
        probabilities_copy[i] = 0.0
    return samples


class DatasetWord2Vec(Dataset):
    """
    This is a custom implementation of the dataset. As we generate more than one sample per line
    in the text file we cannot use the dataset_filelines. We also do not use more than one thread
    to read the data. And the data is shuffled in the batch join operation
    """
    def __init__(self, vocabulary_size=VOCABULARY_SIZE,
                 window_adjacent_words=W2V_WINDOW_ADJACENT_WORDS,
                 close_words_size=W2V_CLOSE_WORDS_SIZE,
                 window_close_words=W2V_WINDOW_CLOSE_WORDS):
        self._size = None
        self.window_adjacent_words = window_adjacent_words
        self.close_words_size = close_words_size
        self.window_close_words = window_close_words
        filename = 'word2vec_dataset_{}'.format(vocabulary_size)
        self.data_file = os.path.join(DIR_DATA_WORD2VEC, filename)

        _, _, word_frequency_dict = load_word2vec_data('word2vec_dataset',
                                                       vocabulary_size=vocabulary_size)
        self.probabilities_dict = {}
        unknown_count = 0
        for k, v in word_frequency_dict.items():
            if k != 0:
                self.probabilities_dict[k] = -math.log(v)
            else:
                unknown_count += v
        self.probabilities_dict[0] = -math.log(unknown_count)

        self.samples_generator = self._samples_generator()

    def _count_num_records(self):
        size = 0
        with open(self.data_file) as f:
            for line in f:
                words = line.split()
                len_text_line = len(words)
                for i, word in enumerate(words):
                    aw_min = max(0, i - self.window_adjacent_words)
                    aw_max = min(len_text_line, i + self.window_adjacent_words + 1)
                    adjacent_words = (i - aw_min) + (aw_max - (i + 1))
                    nsw_min = max(0, min(aw_min, i - self.window_close_words))
                    nsw_max = min(len_text_line, max(aw_max, i + self.window_close_words + 1))
                    close_words = (aw_min - nsw_min) + (nsw_max - aw_max)
                    total_pairs = adjacent_words + min(close_words, self.close_words_size)
                    size += total_pairs
        return size

    def get_size(self):
        if self._size is None:
            self._size = self._count_num_records()
        return self._size

    def _samples_generator(self):
        while True:
            with open(self.data_file) as f:
                for l in f:
                    text_line = [int(w) for w in l.split()]
                    probabilities_tl = [self.probabilities_dict[w] for w in text_line]
                    len_text_line = len(text_line)
                    for i, word in enumerate(text_line):
                        aw_min = max(0, i - self.window_adjacent_words)
                        aw_max = min(len_text_line, i + self.window_adjacent_words + 1)
                        adjacent_words = text_line[aw_min:i] + text_line[i + 1:aw_max]

                        nsw_min = max(0, min(aw_min, i - self.window_close_words))
                        nsw_max = min(len_text_line, max(aw_max, i + self.window_close_words + 1))
                        close_words = text_line[nsw_min:aw_min] + text_line[aw_max:nsw_max]

                        prob = probabilities_tl[nsw_min:aw_min] + probabilities_tl[aw_max:nsw_max]
                        close_words_selected = select_random_labels(close_words,
                                                                    self.close_words_size, prob)

                        context = adjacent_words + close_words_selected
                        for label in context:
                            yield np.int32(label), np.int32(word)

    def py_func_parse_example(self):
        return self.samples_generator.next()

    def read(self, batch_size, shuffle=False):
        inputs_outputs = []

        with tf.device('/cpu'):
            inputs, outputs = tf.py_func(self.py_func_parse_example, [], [tf.int32, tf.int32],
                                         stateful=True, name='parse_example')
        inputs = tf.reshape(inputs, [1])
        outputs = tf.reshape(outputs, [1])
        inputs_outputs.append([inputs, outputs])

        if shuffle:
            capacity = 100 * batch_size
            result = tf.train.shuffle_batch_join(inputs_outputs, batch_size=batch_size,
                                                 min_after_dequeue=capacity / 2, capacity=capacity)
        else:
            capacity = 10 * batch_size
            result = tf.train.batch_join(inputs_outputs, batch_size=batch_size, capacity=capacity)

        return result[0], result[1]


class MyTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, epochs=W2V_EPOCHS):
        self.dataset = dataset
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        max_steps = epochs * dataset.get_size()
        super(MyTrainer, self).__init__(DIR_W2V_LOGDIR, max_steps=max_steps,
                                        monitored_training_session_config=config)

    def model(self,
              batch_size=W2V_BATCH_SIZE,
              vocabulary_size=VOCABULARY_SIZE,
              embedding_size=EMBEDDINGS_SIZE,
              num_negative_samples=W2V_NEGATIVE_NUM_SAMPLES,
              learning_rate_initial=W2V_LEARNING_RATE_INITIAL,
              learning_rate_decay=W2V_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=W2V_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        self.input_label, self.output_word = self.dataset.read(batch_size, shuffle=True)
        input_label_reshaped = tf.reshape(self.input_label, [batch_size])
        output_word_reshaped = tf.reshape(self.output_word, [batch_size, 1])

        # embeddings
        matrix_dimension = [vocabulary_size, embedding_size]
        self.embeddings = tf.Variable(tf.random_uniform(matrix_dimension, -1.0, 1.0),
                                      name='embeddings')
        embed = tf.nn.embedding_lookup(self.embeddings, input_label_reshaped)

        # NCE loss
        stddev = 1.0 / math.sqrt(embedding_size)
        nce_weights = tf.Variable(tf.truncated_normal(matrix_dimension, stddev=stddev))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                  labels=output_word_reshaped,
                                  inputs=embed, num_sampled=num_negative_samples,
                                  num_classes=vocabulary_size)
        self.loss = tf.reduce_mean(nce_loss)

        # learning rate & optimizer
        self.learning_rate = tf.train.exponential_decay(learning_rate_initial, self.global_step,
                                                        learning_rate_decay_steps,
                                                        learning_rate_decay,
                                                        staircase=True, name='learning_rate')
        sgd = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = sgd.minimize(self.loss, global_step=self.global_step)

        # summaries and moving average
        ema = tf.train.ExponentialMovingAverage(0.9)
        ema_assign = ema.apply([self.loss])
        with tf.control_dependencies([self.optimizer]):
            self.training_op = tf.group(ema_assign)
        self.average_loss = ema.average(self.loss)
        tf.summary.scalar('loss', self.loss)
        # saver to save the model
        self.saver = tf.train.Saver()
        # check a nan value in the loss
        self.loss = tf.check_numerics(self.loss, 'loss is nan')

        # embeddings
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        filename_tsv = '{}_{}.tsv'.format('word2vec_dataset', vocabulary_size)
        shutil.copy(os.path.join(DIR_DATA_WORD2VEC, filename_tsv), self.log_dir)
        embedding.metadata_path = filename_tsv
        summary_writer = tf.summary.FileWriter(self.log_dir)
        projector.visualize_embeddings(summary_writer, config)

        # in case you want to get the embeddings from the graph:
        # norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        # self.normalized_embeddings = self.embeddings / norm

        return None

    def create_graph(self):
        return self.model()

    def train_step(self, session, graph_data):
        lr, _, _, loss_val, step = session.run([self.learning_rate, self.training_op, self.loss,
                                                self.average_loss, self.global_step])
        if step % 1000 == 0:
            elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
            m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
            print(m.format(step, loss_val, lr, elapsed_time))

    def after_create_session(self, session, coord):
        self.init_time = time.time()

    def end(self, session):
        print('Saving embeddings in text format...')
        embeddings_eval = self.embeddings.eval(session=session)
        norm = np.sqrt(np.sum(np.square(embeddings_eval)))
        normalized_embeddings = embeddings_eval / norm
        embeddings_file = 'embeddings_{}_{}'.format(VOCABULARY_SIZE, EMBEDDINGS_SIZE)
        with open(os.path.join(DIR_DATA_WORD2VEC, embeddings_file), 'wb') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(normalized_embeddings)


if __name__ == '__main__':
    # start the training
    MyTrainer(dataset=DatasetWord2Vec()).train()
