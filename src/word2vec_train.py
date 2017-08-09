import tensorflow as tf
import math
import numpy as np
import random
import csv
import time
from datetime import timedelta
from tensorport_template import trainer
from tensorflow.python.training import training_util
from word2vec_process_data import load_word2vec_data
from configuration import *


def generate_training_samples(text_lines, word_frequency_dict, epochs=W2V_EPOCHS,
                              window_adjacent_words=W2V_WINDOW_ADJACENT_WORDS,
                              close_words_size=W2V_CLOSE_WORDS_SIZE,
                              window_close_words=W2V_WINDOW_CLOSE_WORDS):
    """
    Generates training samples from the text lines. For every target word a context is selected,
    the adjacent words are added to the context and some random words close to the target are also
    added. The random words are selected by based on their frequency in the text, less frequent
    words are selected more often in order to balance the dataset. This random selection follows a
    log distribution, the values are transformed as:
     p_i = -log(frequence_i) / sum( -log(frequence) )
    :param List[List[int]] text_lines: text lines with ids instead of words
    :param word_frequency_dict: dictionary with the frequency of the words
    :param epochs: number of epochs to generate
    :param window_adjacent_words: adjacent window to select words for the context
    :param close_words_size: number of close words to select
    :param window_close_words: close window to select words for the context
    :return (str,str): pairs of (label,word)
    """
    probabilities_dict = {}
    for k, v in word_frequency_dict.items():
        probabilities_dict[k] = -math.log(v)
    for _ in range(epochs):
        random.shuffle(text_lines)
        for text_line in text_lines:
            probabilities_tl = [probabilities_dict[w] for w in text_line]
            len_text_line = len(text_line)
            for i, word in enumerate(text_line):
                aw_min = max(0, i - window_adjacent_words)
                aw_max = min(len_text_line, i + window_adjacent_words + 1)
                adjacent_words = text_line[aw_min:i] + text_line[i + 1:aw_max]

                nsw_min = max(0, min(aw_min, i - window_close_words))
                nsw_max = min(len_text_line, max(aw_max, i + window_close_words + 1))
                close_words = text_line[nsw_min:aw_min] + text_line[aw_max:nsw_max]

                prob = probabilities_tl[nsw_min:aw_min] + probabilities_tl[aw_max:nsw_max]
                close_words_selected = select_random_labels(close_words, close_words_size, prob)

                context = adjacent_words + close_words_selected
                for label in context:
                    yield label, word


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


def data_generator_buffered(data_generator, buffer_size=100000, randomize=True):
    """
    Creates a buffer of samples generated and returns random elements of this buffer if
    randomize is true.
    :param data_generator: generator of samples
    :param buffer_size: size of the buffer of generated samples
    :param randomize: whether to return a random sample of the buffer or the oldest one
    :return: a sample from the buffer of samples
    """
    buffer = []
    try:
        while len(buffer) < buffer_size:
            buffer.append(data_generator.next())
    except StopIteration:
        pass
    while len(buffer) > 1:
        if randomize:
            random_pos = random.randrange(len(buffer))
        else:
            random_pos = 0
        yield buffer[random_pos]
        del buffer[random_pos]
        try:
            buffer.append(data_generator.next())
        except StopIteration:
            pass
    yield buffer[0]


def generate_batch(data_generator, batch_size):
    """
    Generates batchs of samples given a generator that only generates one sample per iteration
    :param data_generator: generator of samples
    :param batch_size: batch size of the data generated
    :return: a batch list of samples
    """
    batch = []
    for sample in data_generator:
        batch.append(sample)
        if len(batch) == batch_size:
            inputs, outputs = zip(*batch)
            yield np.asarray(inputs), np.asarray(outputs)
            batch = []


class MyTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self):
        dataset_spec = trainer.DatasetSpec('word2vec_train', DIR_DATA_WORD2VEC, LOCAL_REPO,
                                           LOCAL_REPO_W2V_PATH)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(MyTrainer, self).__init__(dataset_spec, DIR_W2V_LOGDIR,
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

        # inputs
        self.input_label = tf.placeholder(tf.int32, shape=[batch_size])
        self.output_word = tf.placeholder(tf.int32, shape=[batch_size])
        input_label_reshaped = tf.reshape(self.input_label, [batch_size])
        output_word_reshaped = tf.reshape(self.output_word, [batch_size, 1])

        # embbedings
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
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

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

        return None

    def train_step(self, session, graph_data):
        # we ignore the parent loop and do all the iterations here
        try:
            for words, labels in generate_batch(data_generator, W2V_BATCH_SIZE):
                lr, _, _, loss_val, step = session.run([self.learning_rate, self.optimizer,
                                                        self.loss, self.average_loss,
                                                        self.global_step],
                                                       feed_dict={
                                                           self.input_label: labels,
                                                           self.output_word: words,
                                                       })
                if step % 100000 == 0:
                    elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
                    m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {:0.1f}'
                    print(m.format(step, loss_val, lr, elapsed_time))
        except StopIteration:
            pass

    def preload_model(self, session, graph_data):
        # restore latest checkpoint
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt:
            self.saver.restore(session, ckpt.model_checkpoint_path)
        self.init_time = time.time()

    def save_embeddings(self, session, graph_data):
        print('Saving embeddings in text format...')
        embeddings_eval = self.embeddings.eval()
        norm = np.sqrt(np.sum(np.square(embeddings_eval)))
        normalized_embeddings = embeddings_eval / norm
        embeddings_file = 'embeddings_{}'.format(EMBEDDINGS_SIZE)
        with open(os.path.join(DIR_DATA_WORD2VEC, embeddings_file), 'wb') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(normalized_embeddings)


if __name__ == '__main__':
    print('Loading dataset...')
    symbols_dict, encoded_text, word_frequency_dict = load_word2vec_data('word2vec_dataset')
    vocabulary_size = len(set(symbols_dict.values()))

    print('Buffering data...')
    data_generator = generate_training_samples(encoded_text, word_frequency_dict)
    data_generator = data_generator_buffered(data_generator)

    # start the training
    trainer = MyTrainer()
    trainer.train(create_graph_fn=trainer.model,
                  train_step_fn=trainer.train_step,
                  pre_train_fn=trainer.preload_model,
                  post_train_fn=trainer.save_embeddings)
