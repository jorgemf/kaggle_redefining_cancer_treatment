import tensorflow as tf
import time
from datetime import timedelta
import numpy as np
import random
from tensorflow.python.training import training_util
from tensorflow.contrib import layers
from . import trainer, metrics
from .tf_dataset_generator import TFDataSetGenerator
from .configuration import *
from .text_classification_train import _load_embeddings


class DocPredictionDataset(TFDataSetGenerator):
    """
    Custom dataset. We use it to feed the session.run method directly.
    """

    def __init__(self, type='train',
                 vocabulary_size=VOCABULARY_SIZE,
                 embedding_size=EMBEDDINGS_SIZE,
                 balance_classes=False):
        self.type = type
        if type != 'train' and type != 'test' and type != 'stage2_test' and type != 'val':
            raise ValueError('Type must be train, test, stage2_test or val')
        docs_filename = '{}_set'.format(type)
        if type == 'train':
            embeds_filename = 'doc_embeddings_{}_{}'.format(vocabulary_size, embedding_size)
        else:
            embeds_filename = 'doc_eval_embeddings_{}_{}_{}'.format(type, vocabulary_size,
                                                                    embedding_size)
        word_embeds = _load_embeddings(vocabulary_size, embedding_size,
                                       filename_prefix='word_embeddings',
                                       from_dir=DIR_DATA_DOC2VEC)

        self.docs_file = os.path.join(DIR_DATA_DOC2VEC, docs_filename)
        self.embeds_file = os.path.join(DIR_DATA_DOC2VEC, embeds_filename)
        self.doc_genes = []
        self.doc_variants = []
        if type == 'train' or type == 'val':
            self.doc_labels = []
        else:
            self.doc_labels = None
        # pre load data in memory for the generator
        with open(self.docs_file) as f:
            for line in f.readlines():
                sp = line.split('||')
                if type == 'train' or type == 'val':
                    self.doc_labels.append(int(sp[0].strip()))
                gene = word_embeds[int(sp[1].strip())]
                self.doc_genes.append(gene)
                variant = np.zeros(embedding_size, dtype=np.float32)
                v = [int(x) for x in sp[2].split()]
                for x in v:
                    variant += np.asarray(word_embeds[x], dtype=np.float32)
                variant /= len(v)
                self.doc_variants.append(variant)

        output_types = (tf.float32, tf.float32, tf.float32, tf.int32)
        with open(self.embeds_file) as f:
            self.embeds = [l.split(',') for l in f.readlines()]
            for i, line in enumerate(self.embeds):
                self.embeds[i] = [float(e) for e in line]
        if type == 'train' and balance_classes:
            self._balance_classes()

        super(DocPredictionDataset, self).__init__(name=type,
                                                   generator=self._generator,
                                                   output_types=output_types,
                                                   min_queue_examples=1000,
                                                   shuffle_size=2000)

    def _balance_classes(self):
        dataset = zip(self.doc_labels, self.embeds)
        classes_group = {}
        for d in dataset:
            if d[0] not in classes_group:
                classes_group[d[0]] = []
            classes_group[d[0]].append(d)
        max_in_class = np.max([len(v) for v in classes_group.values()]) * 2
        new_dataset = []
        for key, class_list in classes_group.iteritems():
            random.shuffle(class_list)
            for index in range(max_in_class - len(class_list)):
                class_list.append(class_list[index])
            new_dataset.extend(class_list)

        random.shuffle(new_dataset)
        self.doc_labels, self.embeds = zip(*new_dataset)

    def _generator(self):
        for i in range(len(self.embeds)):
            embeds = np.asarray(self.embeds[i], dtype=np.float32)
            gene = np.asarray(self.doc_genes[i], dtype=np.float32)
            variant = np.asarray(self.doc_variants[i], dtype=np.float32)
            if self.doc_labels is None:
                # evaluation set:
                yield embeds, gene, variant, np.int32(-1)
            else:
                # train or validation set:
                # subtract 1 to class as classes goes from 1 to 9 (both inclusive)
                label = np.int32(int(self.doc_labels[i]) - 1)
                yield embeds, gene, variant, label


def doc2vec_prediction_model(input_vectors, input_gene, input_variation, output_label, batch_size,
                             is_training, embedding_size, output_classes):
    # inputs/outputs
    input_vectors = tf.reshape(input_vectors, [batch_size, embedding_size])
    input_gene = tf.reshape(input_gene, [batch_size, embedding_size])
    input_variation = tf.reshape(input_variation, [batch_size, embedding_size])
    targets = None
    if output_label is not None:
        output_label = tf.reshape(output_label, [batch_size, 1])
        targets = tf.one_hot(output_label, axis=-1, depth=output_classes, on_value=1.0,
                             off_value=0.0)
        targets = tf.squeeze(targets, axis=1)

    net = tf.concat([input_vectors, input_gene, input_variation], axis=1)
    net = layers.fully_connected(net, embedding_size * 2, activation_fn=tf.nn.relu)
    net = layers.dropout(net, keep_prob=0.85, is_training=is_training)
    net = layers.fully_connected(net, embedding_size, activation_fn=tf.nn.relu)
    net = layers.dropout(net, keep_prob=0.85, is_training=is_training)
    net = layers.fully_connected(net, embedding_size // 4, activation_fn=tf.nn.relu)
    logits = layers.fully_connected(net, output_classes, activation_fn=None)

    return logits, targets


class DocPredictionTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(DocPredictionTrainer, self).__init__(DIR_D2V_DOC_LOGDIR, dataset=dataset,
                                                   monitored_training_session_config=config,
                                                   save_checkpoint_secs=60,
                                                   log_step_count_steps=1000,
                                                   save_summaries_steps=1000)

    def model(self,
              input_vectors, input_gene, input_variation, output_label, batch_size,
              embedding_size=EMBEDDINGS_SIZE,
              output_classes=9,
              learning_rate_initial=D2V_DOC_LEARNING_RATE_INITIAL,
              learning_rate_decay=D2V_DOC_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=D2V_DOC_LEARNING_RATE_DECAY_STEPS):
        self.global_step = training_util.get_or_create_global_step()

        logits, targets = doc2vec_prediction_model(input_vectors, input_gene, input_variation,
                                                   output_label, batch_size,
                                                   is_training=True, embedding_size=embedding_size,
                                                   output_classes=output_classes)

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

    def create_graph(self, dataset_tensor, batch_size):
        input_vectors, input_gene, input_variation, output_label = dataset_tensor
        return self.model(input_vectors, input_gene, input_variation, output_label, batch_size)

    def step(self, session, graph_data):
        lr, _, loss, step, metrics = \
            session.run([self.learning_rate, self.optimizer, self.loss,
                         self.global_step, self.metrics])
        if self.is_chief and time.time() > self.print_timestamp + 1 * 60:
            self.print_timestamp = time.time()
            elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
            m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}  ' \
                'precision: {}  recall: {}  accuracy: {}'
            logging.info(m.format(step, loss, lr, elapsed_time,
                                  metrics['precision'], metrics['recall'], metrics['accuracy']))

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()


if __name__ == '__main__':
    import logging

    logging.getLogger().setLevel(logging.INFO)
    # start the training
    trainer = DocPredictionTrainer(dataset=DocPredictionDataset())
    trainer.run(epochs=D2V_DOC_EPOCHS, batch_size=D2V_DOC_BATCH_SIZE)
