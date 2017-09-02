import tensorflow as tf
import math
import numpy as np
import random
import csv
import time
from datetime import timedelta
import shutil
import logging
from tensorflow.python.training import training_util
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import layers
from src import trainer
from src.word2vec_train import generate_batch, data_generator_buffered
from src.configuration import *


class Doc2VecDataset(object):
    """
    Custom dataset. We use it to feed the session.run method directly.
    """

    def __init__(self, type='train',
                 shuffle_data=True,
                 context_size=D2V_CONTEXT_SIZE,
                 batch_size=D2V_BATCH_SIZE):
        self._size = None
        self.context_size = context_size
        filename = '{}_set'.format(type)
        self.data_file = os.path.join(DIR_DATA_DOC2VEC, filename)

        samples_generator = self._samples_generator()
        samples_generator = data_generator_buffered(samples_generator, randomize=shuffle_data)
        self.batch_generator = generate_batch(samples_generator, batch_size)

    def _count_num_records(self):
        size = 0
        with open(self.data_file) as f:
            for line in f:
                size += len(line.split()) - self.context_size - 1  # minus context and real class
        return size

    def get_size(self):
        if self._size is None:
            self._size = self._count_num_records()
        return self._size

    def _samples_generator(self):
        with open(self.data_file) as f:
            lines = [l.split()[1:] for l in f.readlines()]  # read lines and skip the class
            lines = [l for l in lines if len(l) > self.context_size]  # skip shorter lines
            for i, line in enumerate(lines):
                lines[i] = [int(w) for w in line]
        indixes = [0] * len(lines)
        while True:
            for doc_id in range(len(lines)):
                line = lines[doc_id]
                if indixes[doc_id] >= len(line) - self.context_size - 1:
                    indixes[doc_id] = 0
                index = indixes[doc_id]
                context = line[index:index + self.context_size]
                label = line[index + self.context_size + 1]
                yield (doc_id, context), label

    def read(self):
        inputs, label = self.batch_generator.next()
        doc_id, context = zip(*inputs)
        return context, doc_id, label


class Doc2VecTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, epochs=D2V_EPOCHS):
        self.dataset = dataset
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        max_steps = epochs * dataset.get_size()
        super(Doc2VecTrainer, self).__init__(DIR_D2V_LOGDIR, max_steps=max_steps,
                                             monitored_training_session_config=config,
                                             log_step_count_steps=1000, save_summaries_steps=1000)

    def model(self,
              batch_size=D2V_BATCH_SIZE,
              vocabulary_size=VOCABULARY_SIZE,
              embedding_size=EMBEDDINGS_SIZE,
              docs_size=D2_TRAIN_DOCS_SIZE,
              context_size=D2V_CONTEXT_SIZE,
              num_negative_samples=D2V_NEGATIVE_NUM_SAMPLES,
              learning_rate_initial=D2V_LEARNING_RATE_INITIAL,
              learning_rate_decay=D2V_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=D2V_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        self.input_words = tf.placeholder(tf.int32, shape=[batch_size, context_size],
                                          name='input_context')
        self.input_doc = tf.placeholder(tf.int32, shape=[batch_size], name='input_doc')
        self.output_label = tf.placeholder(tf.int32, shape=[batch_size], name='output_label')
        output_label = tf.reshape(self.output_label, [batch_size, 1])

        # embeddings
        self.word_embeddings = tf.get_variable(shape=[vocabulary_size, embedding_size],
                                               initializer=layers.xavier_initializer(),
                                               dtype=tf.float32, name='word_embeddings')
        self.doc_embeddings = tf.get_variable(shape=[docs_size, embedding_size],
                                              initializer=layers.xavier_initializer(),
                                              dtype=tf.float32, name='doc_embeddings')
        words_embed = tf.nn.embedding_lookup(self.word_embeddings, self.input_words)
        doc_embed = tf.nn.embedding_lookup(self.word_embeddings, self.input_doc)
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

        # embeddings
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.word_embeddings.name
        filename_tsv = '{}_{}.tsv'.format('word2vec_dataset', vocabulary_size)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        shutil.copy(os.path.join(DIR_DATA_WORD2VEC, filename_tsv), self.log_dir)
        embedding.metadata_path = filename_tsv
        summary_writer = tf.summary.FileWriter(self.log_dir)
        projector.visualize_embeddings(summary_writer, config)

        # normalize the embeddings to save them
        norm_word = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
        self.normalized_word_embeddings = self.word_embeddings / norm_word
        norm_doc = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
        self.normalized_doc_embeddings = self.doc_embeddings / norm_doc

        return None

    def create_graph(self):
        return self.model()

    def train_step(self, session, graph_data):
        words, doc, label = self.dataset.read()
        if self.is_chief:
            lr, _, loss, step, embeddings_words, embeddings_docs = \
                session.run([self.learning_rate, self.optimizer, self.loss, self.global_step,
                             self.normalized_word_embeddings, self.normalized_doc_embeddings],
                            feed_dict={
                                self.input_words: words,
                                self.input_doc: doc,
                                self.output_label: label,
                            })
            if step % 10000 == 0:
                elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
                m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
                print(m.format(step, loss, lr, elapsed_time))
                current_time = time.time()
                embeddings_file = 'word_embeddings_{}_{}'.format(VOCABULARY_SIZE, EMBEDDINGS_SIZE)
                embeddings_filepath = os.path.join(DIR_DATA_DOC2VEC, embeddings_file)
                if not os.path.exists(embeddings_filepath):
                    self.save_embeddings(embeddings_words, embeddings_docs)
                else:
                    embeddings_file_timestamp = os.path.getmtime(embeddings_filepath)
                    # save the embeddings every 30 minutes
                    if current_time - embeddings_file_timestamp > 30 * 60:
                        self.save_embeddings(embeddings_words, embeddings_docs)
        else:
            session.run([self.optimizer],
                        feed_dict={
                            self.input_words: words,
                            self.input_doc: doc,
                            self.output_label: label,
                        })

    def after_create_session(self, session, coord):
        self.init_time = time.time()

    def end(self, session):
        if self.is_chief:
            try:
                embeddings_words, embeddings_docs = \
                    session.run([self.normalized_word_embeddings, self.normalized_doc_embeddings])
            except:
                words, doc, label = self.dataset.read()
                embeddings_words, embeddings_docs = \
                    session.run([self.normalized_word_embeddings, self.normalized_doc_embeddings],
                                feed_dict={
                                    self.input_words: words,
                                    self.input_doc: doc,
                                    self.output_label: label,
                                })
            self.save_embeddings(embeddings_words, embeddings_docs)

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
    Doc2VecTrainer(dataset=Doc2VecDataset()).train()
