import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
from tensorflow.python.training import training_util
from tensorflow.contrib import layers
import src.trainer as trainer
from src.tf_dataset_generator import TFDataSetGenerator
from src.configuration import *
import src.metrics as metrics


class DocPredictionDataset(TFDataSetGenerator):
    """
    Custom dataset. We use it to feed the session.run method directly.
    """

    def __init__(self, type='train',
                 batch_size=D2V_DOC_BATCH_SIZE,
                 vocabulary_size=VOCABULARY_SIZE,
                 embedding_size=EMBEDDINGS_SIZE):
        self.batch_size = batch_size
        docs_filename = '{}_set'.format(type)
        embeds_filename = 'doc_embeddings_{}_{}'.format(vocabulary_size, embedding_size)
        if type == 'test':
            embeds_filename = '{}_{}'.format(type, embeds_filename)

        self.docs_file = os.path.join(DIR_DATA_DOC2VEC, docs_filename)
        self.embeds_file = os.path.join(DIR_DATA_DOC2VEC, embeds_filename)

        # pre load data in memory for the generator
        with open(self.docs_file) as f:
            self.doc_labels = [int(line.split()[0]) for line in f.readlines()]
        with open(self.embeds_file) as f:
            self.embeds = [l.split(',') for l in f.readlines()]
            for i, line in enumerate(self.embeds):
                self.embeds[i] = [float(e) for e in line]

        output_types = (tf.float32, tf.int32)
        super(DocPredictionDataset, self).__init__(name=type,
                                                   generator=self._generator,
                                                   output_types=output_types,
                                                   min_queue_examples=1000,
                                                   shuffle_size=2000)

    def _generator(self):
        # TODO while repeat doesn't work on dataset do it here:
        for _ in range(self.epochs):
            for i in range(len(self.embeds)):
                yield np.asarray(self.embeds[i], dtype=np.float32), np.int32(self.doc_labels[i])


class DocPredictionTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, epochs=D2V_DOC_EPOCHS, batch_size=D2V_DOC_BATCH_SIZE):
        # TODO while repeat doesn't work on dataset do it in the dataset:
        dataset.epochs = epochs
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(DocPredictionTrainer, self).__init__(DIR_D2V_DOC_LOGDIR,
                                                   monitored_training_session_config=config,
                                                   log_step_count_steps=1000,
                                                   save_summaries_steps=1000)

    def model(self,
              input_vectors, output_label,
              embedding_size=EMBEDDINGS_SIZE,
              output_classes=9,
              learning_rate_initial=D2V_DOC_LEARNING_RATE_INITIAL,
              learning_rate_decay=D2V_DOC_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=D2V_DOC_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        # inputs/outputs
        input_vectors = tf.reshape(input_vectors, [self.batch_size, embedding_size])
        output_label = tf.reshape(output_label, [self.batch_size, 1])
        targets = tf.one_hot(output_label, axis=-1, depth=output_classes, on_value=1.0,
                             off_value=0.0)
        targets = tf.squeeze(targets)

        net = layers.fully_connected(input_vectors, embedding_size // 2, activation_fn=tf.tanh)
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
        next_tensor = self.dataset.read(batch_size=self.batch_size,
                                        num_epochs=self.epochs,
                                        shuffle=True,
                                        task_spec=self.task_spec)
        input_vectors, output_label = next_tensor
        return self.model(input_vectors, output_label)

    def step(self, session, graph_data):
        lr, _, loss, step, metrics = \
            session.run([self.learning_rate, self.optimizer, self.loss,
                         self.global_step, self.metrics])
        if self.is_chief and time.time() > self.print_timestamp + 5 * 60*0:
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
    DocPredictionTrainer(dataset=DocPredictionDataset()).run()
