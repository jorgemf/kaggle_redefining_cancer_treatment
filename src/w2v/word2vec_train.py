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
from tensorflow.contrib import layers
from .. import trainer
from ..tf_dataset_generator import TFDataSetGenerator
from ..w2v.word2vec_process_data import load_word2vec_data
from ..configuration import *


class Word2VecDataset(TFDataSetGenerator):
    def __init__(self, vocabulary_size=VOCABULARY_SIZE,
                 window_adjacent_words=W2V_WINDOW_ADJACENT_WORDS,
                 close_words_size=W2V_CLOSE_WORDS_SIZE, window_close_words=W2V_WINDOW_CLOSE_WORDS):
        filename = 'word2vec_dataset_{}'.format(vocabulary_size)
        self.data_file = os.path.join(DIR_DATA_WORD2VEC, filename)
        self.window_adjacent_words = window_adjacent_words
        self.close_words_size = close_words_size
        self.window_close_words = window_close_words

        _, _, word_frequency_dict = load_word2vec_data('word2vec_dataset',
                                                       vocabulary_size=vocabulary_size)
        self.probabilities_dict = { }
        unknown_count = 0
        for k, v in word_frequency_dict.items():
            if k != 0:
                self.probabilities_dict[k] = -math.log(v)
            else:
                unknown_count += v
        self.probabilities_dict[0] = -math.log(unknown_count)
        output_types = (tf.int32, tf.int32)
        super(Word2VecDataset, self).__init__(name='train', generator=self._generator,
                                              output_types=output_types, min_queue_examples=1000,
                                              shuffle_size=100000)

    def _generator(self):
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
                    close_words_selected = self._select_random_labels(close_words,
                                                                      self.close_words_size, prob)

                    context = adjacent_words + close_words_selected
                    for label in context:
                        yield np.int32(label), np.int32(word)

    def _select_random_labels(self, labels, num_labels, probabilities):
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


class Word2VecTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(Word2VecTrainer, self).__init__(DIR_W2V_LOGDIR, dataset=dataset,
                                              monitored_training_session_config=config,
                                              log_step_count_steps=1000, save_summaries_steps=1000)

    def model(self, input_label, output_word, batch_size, vocabulary_size=VOCABULARY_SIZE,
              embedding_size=EMBEDDINGS_SIZE, num_negative_samples=W2V_NEGATIVE_NUM_SAMPLES,
              learning_rate_initial=W2V_LEARNING_RATE_INITIAL,
              learning_rate_decay=W2V_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=W2V_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        input_label_reshaped = tf.reshape(input_label, [batch_size])
        output_word_reshaped = tf.reshape(output_word, [batch_size, 1])

        # embeddings
        matrix_dimension = [vocabulary_size, embedding_size]
        self.embeddings = tf.get_variable(shape=matrix_dimension,
                                          initializer=layers.xavier_initializer(), dtype=tf.float32,
                                          name='embeddings')
        embed = tf.nn.embedding_lookup(self.embeddings, input_label_reshaped)

        # NCE loss
        stddev = 1.0 / math.sqrt(embedding_size)
        nce_weights = tf.Variable(tf.truncated_normal(matrix_dimension, stddev=stddev))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                  labels=output_word_reshaped, inputs=embed,
                                  num_sampled=num_negative_samples, num_classes=vocabulary_size)
        self.loss = tf.reduce_mean(nce_loss)
        tf.summary.scalar('loss', self.loss)

        # learning rate & optimizer
        self.learning_rate = tf.train.exponential_decay(learning_rate_initial, self.global_step,
                                                        learning_rate_decay_steps,
                                                        learning_rate_decay, staircase=True,
                                                        name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)
        sgd = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = sgd.minimize(self.loss, global_step=self.global_step)

        # saver to save the model
        self.saver = tf.train.Saver()
        # check a nan value in the loss
        self.loss = tf.check_numerics(self.loss, 'loss is nan')

        # embeddings
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        filename_tsv = '{}_{}.tsv'.format('word2vec_dataset', vocabulary_size)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        shutil.copy(os.path.join(DIR_DATA_WORD2VEC, filename_tsv), self.log_dir)
        embedding.metadata_path = filename_tsv
        summary_writer = tf.summary.FileWriter(self.log_dir)
        projector.visualize_embeddings(summary_writer, config)

        # normalize the embeddings to save them
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm

        return None

    def create_graph(self, dataset_tensor, batch_size):
        input_label, output_word = dataset_tensor
        return self.model(input_label, output_word, batch_size)

    def step(self, session, graph_data):
        if self.is_chief:
            lr, _, loss, step, self.embeddings = session.run(
                    [self.learning_rate, self.optimizer, self.loss, self.global_step,
                     self.normalized_embeddings])
            if time.time() > self.print_timestamp + 5 * 60:
                self.print_timestamp = time.time()
                elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
                m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
                print(m.format(step, loss, lr, elapsed_time))
                current_time = time.time()
                embeddings_file = 'embeddings_{}_{}'.format(VOCABULARY_SIZE, EMBEDDINGS_SIZE)
                embeddings_filepath = os.path.join(DIR_DATA_WORD2VEC, embeddings_file)
                if not os.path.exists(embeddings_filepath):
                    self.save_embeddings(self.embeddings)
                else:
                    embeddings_file_timestamp = os.path.getmtime(embeddings_filepath)
                    # save the embeddings every 30 minutes
                    if current_time - embeddings_file_timestamp > 30 * 60:
                        self.save_embeddings(self.embeddings)
        else:
            session.run([self.optimizer])

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()

    def end(self, session):
        if self.is_chief:
            self.save_embeddings(self.embeddings)

    def save_embeddings(self, normalized_embeddings):
        print('Saving embeddings in text format...')
        embeddings_file = 'embeddings_{}_{}'.format(VOCABULARY_SIZE, EMBEDDINGS_SIZE)
        embeddings_filepath = os.path.join(DIR_DATA_WORD2VEC, embeddings_file)
        with open(embeddings_filepath, 'w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(normalized_embeddings)
        # copy the embeddings file to the log dir so we can download it from tensorport
        if os.path.exists(embeddings_filepath):
            shutil.copy(embeddings_filepath, self.log_dir)


if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.INFO)
    # start the training
    trainer = Word2VecTrainer(dataset=Word2VecDataset())
    trainer.run(epochs=W2V_EPOCHS, batch_size=W2V_BATCH_SIZE)
