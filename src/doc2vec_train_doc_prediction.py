import tensorflow as tf
import random
import time
from datetime import timedelta
from tensorflow.python.training import training_util
from tensorflow.contrib import layers
import trainer
from word2vec_train import generate_batch, data_generator_buffered
from configuration import *
import metrics


class DocPredictionDataset(object):
    """
    Custom dataset. We use it to feed the session.run method directly.
    """

    def __init__(self, type='train',
                 shuffle_data=True,
                 batch_size=D2V_DOC_BATCH_SIZE,
                 vocabulary_size=VOCABULARY_SIZE,
                 embedding_size=EMBEDDINGS_SIZE):
        self._size = None
        docs_filename = '{}_set'.format(type)
        embeds_filename = 'doc_embeddings_{}_{}'.format(vocabulary_size, embedding_size)
        if type == 'test':
            embeds_filename = '{}_{}'.format(type, embeds_filename)

        self.docs_file = os.path.join(DIR_DATA_DOC2VEC, docs_filename)
        self.embeds_file = os.path.join(DIR_DATA_DOC2VEC, embeds_filename)

        samples_generator = self._samples_generator()
        samples_generator = data_generator_buffered(samples_generator, randomize=shuffle_data)
        self.batch_generator = generate_batch(samples_generator, batch_size)

    def _count_num_records(self):
        size = 0
        with open(self.docs_file) as f:
            for _ in f:
                size += 1
        return size

    def get_size(self):
        if self._size is None:
            self._size = self._count_num_records()
        return self._size

    def _samples_generator(self):
        with open(self.docs_file) as f:
            doc_labels = [int(line.split()[0]) for line in f.readlines()]
        with open(self.embeds_file) as f:
            embeds = [l.split(',') for l in f.readlines()]
            for i, line in enumerate(embeds):
                embeds[i] = [float(e) for e in line]

        while True:
            r = list(range(self.get_size()))
            random.shuffle(r)
            for i in r:
                yield embeds[i], doc_labels[i]

    def read(self):
        return self.batch_generator.next()


class DocPredictionTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, epochs=D2V_DOC_EPOCHS):
        self.dataset = dataset
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        max_steps = epochs * dataset.get_size()
        super(DocPredictionTrainer, self).__init__(DIR_D2V_DOC_LOGDIR, max_steps=max_steps,
                                                   monitored_training_session_config=config,
                                                   log_step_count_steps=1000,
                                                   save_summaries_steps=1000)

    def model(self,
              batch_size=D2V_DOC_BATCH_SIZE,
              embedding_size=EMBEDDINGS_SIZE,
              output_classes=9,
              learning_rate_initial=D2V_DOC_LEARNING_RATE_INITIAL,
              learning_rate_decay=D2V_DOC_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=D2V_DOC_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        self.input_vectors = tf.placeholder(tf.float32, shape=[batch_size, embedding_size],
                                            name='input_vectors')
        self.output_label = tf.placeholder(tf.int32, shape=[batch_size], name='output_label')
        output_label = tf.reshape(self.output_label, [batch_size, 1])
        targets = tf.one_hot(output_label, axis=-1, depth=output_classes, on_value=1.0,
                             off_value=0.0)

        net = layers.fully_connected(self.input_vectors, embedding_size / 2, activation_fn=tf.tanh)
        logits = layers.fully_connected(net, output_classes, activation_fn=None)

        self.prediction = tf.nn.softmax(logits)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        self.loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('loss', self.loss)

        # learning rate & optimizer
        self.learning_rate = tf.train.exponential_decay(learning_rate_initial, self.global_step,
                                                        learning_rate_decay_steps,
                                                        learning_rate_decay,
                                                        staircase=True, name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)
        sgd = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = sgd.minimize(self.loss, global_step=self.global_step)

        # metrics
        self.metrics = metrics.single_label(self.prediction, targets)

        # saver to save the model
        self.saver = tf.train.Saver()
        # check a nan value in the loss
        self.loss = tf.check_numerics(self.loss, 'loss is nan')

        return None

    def create_graph(self):
        return self.model()

    def train_step(self, session, graph_data):
        embed, label = self.dataset.read()
        lr, _, loss, step, metrics = \
            session.run([self.learning_rate, self.optimizer, self.loss,
                         self.global_step, self.metrics],
                        feed_dict={
                            self.input_vectors: embed,
                            self.output_label: label,
                        })
        if self.is_chief and time.time() > self.print_timestamp + 5 * 60:
            self.print_timestamp = time.time()
            elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
            m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}  ' \
                'precision: {}  recall: {}  accuracy: {}'
            print(m.format(step, loss, lr, elapsed_time,
                           metrics['precision'], metrics['recall'], metrics['accuracy']))

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()


if __name__ == '__main__':
    # start the training
    DocPredictionTrainer(dataset=DocPredictionDataset()).train()
