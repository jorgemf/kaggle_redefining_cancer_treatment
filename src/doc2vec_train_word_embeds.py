import tensorflow as tf
import csv
import time
from datetime import timedelta
import shutil
import numpy as np
from tensorflow.python.training import training_util
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import layers
import trainer
from tf_dataset_generator import TFDataSetGenerator
from configuration import *


class Doc2VecDataset(TFDataSetGenerator):
    """
    Custom dataset. We use it to feed the session.run method directly.
    """

    def __init__(self, type='train',
                 context_size=D2V_CONTEXT_SIZE):
        self.type = type
        self.context_size = context_size
        filename = '{}_set'.format(type)
        with open(self.data_file) as f:
            self.num_docs = len(f.readlines())
        self.data_file = os.path.join(DIR_DATA_DOC2VEC, filename)

        # pre load data in memory for the generator
        with open(self.data_file) as f:
            # read lines and skip the class
            self._data_lines = [l.split()[1:] for l in f.readlines()]
            # skip shorter lines
            self._data_lines = [l for l in self._data_lines if len(l) > self.context_size]
            for i, line in enumerate(self._data_lines):
                self._data_lines[i] = [int(w) for w in line]
        self._data_indixes = [0] * len(self._data_lines)

        output_types = (tf.int32, tf.int32, tf.int32)
        super(Doc2VecDataset, self).__init__(name=type,
                                             generator=self._generator,
                                             output_types=output_types,
                                             min_queue_examples=1000,
                                             shuffle_size=5000)

    def _generator(self):
        for doc_id in range(len(self._data_lines)):
            line = self._data_lines[doc_id]
            if self._data_indixes[doc_id] >= len(line) - self.context_size - 1:
                self._data_indixes[doc_id] = 0
            index = self._data_indixes[doc_id]
            context = line[index:index + self.context_size]
            label = line[index + self.context_size + 1]
            yield np.int32(doc_id), np.int32(context), np.int32(label)


class Doc2VecTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, epochs=D2V_EPOCHS, batch_size=D2V_BATCH_SIZE):
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(Doc2VecTrainer, self).__init__(DIR_D2V_LOGDIR,
                                             monitored_training_session_config=config,
                                             log_step_count_steps=1000, save_summaries_steps=1000)

    def model(self,
              input_words, input_doc, output_label,
              vocabulary_size=VOCABULARY_SIZE,
              embedding_size=EMBEDDINGS_SIZE,
              context_size=D2V_CONTEXT_SIZE,
              num_negative_samples=D2V_NEGATIVE_NUM_SAMPLES,
              learning_rate_initial=D2V_LEARNING_RATE_INITIAL,
              learning_rate_decay=D2V_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=D2V_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        input_words = tf.reshape(input_words, [self.batch_size, context_size])
        input_doc = tf.reshape(input_doc, [self.batch_size])
        output_label = tf.reshape(output_label, [self.batch_size, 1])

        # embeddings
        self.word_embeddings = tf.get_variable(shape=[vocabulary_size, embedding_size],
                                               initializer=layers.xavier_initializer(),
                                               dtype=tf.float32, name='word_embeddings')
        self.doc_embeddings = tf.get_variable(shape=[self.dataset.num_docs, embedding_size],
                                              initializer=layers.xavier_initializer(),
                                              dtype=tf.float32, name='doc_embeddings')
        words_embed = tf.nn.embedding_lookup(self.word_embeddings, input_words)
        doc_embed = tf.nn.embedding_lookup(self.word_embeddings, input_doc)
        # average the words_embeds
        words_embed_average = tf.reduce_mean(words_embed, axis=1)
        embed = tf.concat([words_embed_average, doc_embed], axis=1)

        # NCE loss
        nce_weights = tf.get_variable(shape=[vocabulary_size, embedding_size * 2],
                                      initializer=layers.xavier_initializer(),
                                      dtype=tf.float32, name='nce_weights')
        nce_biases = tf.get_variable(shape=[vocabulary_size],
                                     initializer=layers.xavier_initializer(),
                                     dtype=tf.float32, name='nce_biases')
        nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                  labels=output_label,
                                  inputs=embed, num_sampled=num_negative_samples,
                                  num_classes=vocabulary_size)
        self.loss = tf.reduce_mean(nce_loss)
        tf.summary.scalar('loss', self.loss)

        # learning rate & optimizer
        self.learning_rate = tf.train.exponential_decay(learning_rate_initial, self.global_step,
                                                        learning_rate_decay_steps,
                                                        learning_rate_decay,
                                                        staircase=True, name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)
        sgd = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = sgd.minimize(self.loss, global_step=self.global_step)

        # saver to save the model
        self.saver = tf.train.Saver()
        # check a nan value in the loss
        self.loss = tf.check_numerics(self.loss, 'loss is nan')

        # embeddings in tensorboard
        config = projector.ProjectorConfig()
        # words
        embedding = config.embeddings.add()
        embedding.tensor_name = self.word_embeddings.name
        filename_tsv = '{}_{}.tsv'.format('word2vec_dataset', vocabulary_size)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        shutil.copy(os.path.join(DIR_DATA_WORD2VEC, filename_tsv), self.log_dir)
        embedding.metadata_path = filename_tsv
        # docs
        embedding = config.embeddings.add()
        embedding.tensor_name = self.doc_embeddings.name
        filename_tsv = 'train_set_classes.tsv'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        shutil.copy(os.path.join(DIR_DATA_DOC2VEC, filename_tsv), self.log_dir)
        embedding.metadata_path = filename_tsv

        summary_writer = tf.summary.FileWriter(self.log_dir)
        projector.visualize_embeddings(summary_writer, config)

        return None

    def create_graph(self):
        next_tensor = self.dataset.read(batch_size=self.batch_size,
                                        num_epochs=self.epochs,
                                        shuffle=True,
                                        task_spec=self.task_spec)
        input_word, input_doc, output_label = next_tensor
        return self.model(input_word, input_doc, output_label)

    def train_step(self, session, graph_data):
        if self.is_chief:
            lr, _, loss, step, self.embeddings_words, self.embeddings_docs = \
                session.run([self.learning_rate, self.optimizer, self.loss, self.global_step,
                             self.word_embeddings, self.doc_embeddings])
            if time.time() > self.print_timestamp + 5 * 60:
                self.print_timestamp = time.time()
                elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
                m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
                print(m.format(step, loss, lr, elapsed_time))
                current_time = time.time()
                embeddings_file = 'word_embeddings_{}_{}'.format(VOCABULARY_SIZE, EMBEDDINGS_SIZE)
                embeddings_filepath = os.path.join(DIR_DATA_DOC2VEC, embeddings_file)
                if not os.path.exists(embeddings_filepath):
                    self.save_embeddings(self.embeddings_words, self.embeddings_docs)
                else:
                    embeddings_file_timestamp = os.path.getmtime(embeddings_filepath)
                    # save the embeddings every 30 minutes
                    if current_time - embeddings_file_timestamp > 30 * 60:
                        self.save_embeddings(self.embeddings_words, self.embeddings_docs)
        else:
            session.run([self.optimizer])

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()

    def end(self, session):
        if self.is_chief:
            self.save_embeddings(self.embeddings_words, self.embeddings_docs)

    def save_embeddings(self, word_embeddings, doc_embeddings):
        print('Saving embeddings in text format...')
        for prefix, embeddings in zip(['word', 'doc'], [word_embeddings, doc_embeddings]):
            embeddings_file = '{}_embeddings_{}_{}'.format(prefix, VOCABULARY_SIZE, EMBEDDINGS_SIZE)
            embeddings_filepath = os.path.join(DIR_DATA_DOC2VEC, embeddings_file)
            with open(embeddings_filepath, 'wb') as f:
                writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerows(embeddings)
            # copy the embeddings file to the log dir so we can download it from tensorport
            if os.path.exists(embeddings_filepath):
                shutil.copy(embeddings_filepath, self.log_dir)


if __name__ == '__main__':
    # start the training
    Doc2VecTrainer(dataset=Doc2VecDataset()).run()
