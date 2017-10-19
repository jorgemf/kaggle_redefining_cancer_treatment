import tensorflow as tf
import csv
import time
from datetime import timedelta
import shutil
import sys
from tensorflow.contrib import layers
from .. import trainer
from .doc2vec_train_word_embeds import Doc2VecDataset
from ..rnn.text_classification_train import _load_embeddings
from tensorflow.python.training import training_util
from ..configuration import *


class Doc2VecTrainerEval(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, log_dir=DIR_D2V_EVAL_LOGDIR):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(Doc2VecTrainerEval, self).__init__(log_dir, dataset=dataset,
                                                 monitored_training_session_config=config,
                                                 log_step_count_steps=1000,
                                                 save_summaries_steps=1000)

    def model(self,
              input_doc, input_words, output_label, batch_size,
              vocabulary_size=VOCABULARY_SIZE,
              embedding_size=EMBEDDINGS_SIZE,
              context_size=D2V_CONTEXT_SIZE,
              num_negative_samples=D2V_NEGATIVE_NUM_SAMPLES,
              learning_rate_initial=D2V_LEARNING_RATE_INITIAL,
              learning_rate_decay=D2V_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=D2V_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        input_doc = tf.reshape(input_doc, [batch_size])
        input_words = tf.reshape(input_words, [batch_size, context_size])
        output_label = tf.reshape(output_label, [batch_size, 1])

        # embeddings
        word_embeddings = _load_embeddings(vocabulary_size, embedding_size,
                                           filename_prefix='word_embeddings',
                                           from_dir=DIR_DATA_DOC2VEC)
        self.word_embeddings = tf.constant(value=word_embeddings,
                                           shape=[vocabulary_size, embedding_size],
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
        return None

    def create_graph(self, dataset_tensor, batch_size):
        input_doc, input_word, output_label = dataset_tensor
        return self.model(input_doc, input_word, output_label, batch_size)

    def step(self, session, graph_data):
        if self.is_chief:
            lr, _, loss, step, self.embeddings_docs = \
                session.run([self.learning_rate, self.optimizer, self.loss, self.global_step,
                             self.doc_embeddings])
            if time.time() > self.print_timestamp + 5 * 60:
                self.print_timestamp = time.time()
                elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
                m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
                print(m.format(step, loss, lr, elapsed_time))
                current_time = time.time()
                embeddings_file = 'doc_eval_embeddings_{}_{}_{}'.format(self.dataset.type,
                                                                        VOCABULARY_SIZE,
                                                                        EMBEDDINGS_SIZE)
                embeddings_filepath = os.path.join(DIR_DATA_DOC2VEC, embeddings_file)
                if not os.path.exists(embeddings_filepath):
                    self.save_embeddings(self.embeddings_docs)
                else:
                    embeddings_file_timestamp = os.path.getmtime(embeddings_filepath)
                    # save the embeddings every 30 minutes
                    if current_time - embeddings_file_timestamp > 30 * 60:
                        self.save_embeddings(self.embeddings_docs)
        else:
            session.run([self.optimizer])

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()

    def end(self, session):
        if self.is_chief:
            self.save_embeddings(self.embeddings_docs)

    def save_embeddings(self, doc_embeddings):
        print('Saving embeddings in text format...')
        embeddings_file = 'doc_eval_embeddings_{}_{}_{}'.format(self.dataset.type,
                                                                VOCABULARY_SIZE, EMBEDDINGS_SIZE)
        embeddings_filepath = os.path.join(DIR_DATA_DOC2VEC, embeddings_file)
        with open(embeddings_filepath, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(doc_embeddings)
        # copy the embeddings file to the log dir so we can download it from tensorport
        if os.path.exists(embeddings_filepath):
            shutil.copy(embeddings_filepath, self.log_dir)


if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == 'train_val':
        # start the training for eval
        trainer = Doc2VecTrainerEval(dataset=Doc2VecDataset(type='val'),
                                     log_dir=os.path.join(DIR_D2V_EVAL_LOGDIR, 'val'))
        trainer.run(epochs=D2V_EPOCHS, batch_size=D2V_BATCH_SIZE)
    elif len(sys.argv) > 1 and sys.argv[1] == 'train_test':
        # start the training for second stage dataset
        trainer = Doc2VecTrainerEval(dataset=Doc2VecDataset(type='stage2_test'),
                                     log_dir=os.path.join(DIR_D2V_EVAL_LOGDIR, 'test'))
        trainer.run(epochs=D2V_EPOCHS, batch_size=D2V_BATCH_SIZE)
