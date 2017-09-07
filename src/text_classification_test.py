import tensorflow as tf
import csv
import time
import logging
from datetime import timedelta
import numpy as np
from tensorflow.contrib import slim
from tensorflow.python.training import training_util
from tensorflow.python.training.monitored_session import SingularMonitoredSession
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook
from configuration import *
import trainer
import metrics


class TextClassificationTest(trainer.Trainer):
    """
    Helper class to run the training and create the model for the test. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, text_classification_model, batch_size=TC_BATCH_SIZE,
                 logdir=DIR_TC_LOGDIR):
        self.dataset = dataset
        self.text_classification_model = text_classification_model
        self.batch_size = batch_size
        max_steps = dataset.get_size() / batch_size
        super(TextClassificationTest, self).__init__(logdir, max_steps=max_steps)

    def _load_embeddings(self, vocabulary_size, embeddings_size):
        embeddings = []
        embeddings_file = 'embeddings_{}_{}'.format(vocabulary_size, embeddings_size)
        with open(os.path.join(DIR_DATA_WORD2VEC, embeddings_file), 'r') as file:
            reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                embeddings.append([float(r) for r in row])
        return embeddings

    def run(self):
        logging.info('Creating graph...')
        with tf.Graph().as_default():
            graph_data = self.create_graph()

            logging.info('Creating SingularMonitoredSession...')
            hooks = [self, StopAtStepHook(num_steps=self.max_steps)]
            with SingularMonitoredSession(hooks=hooks) as sess:
                logging.info('Starting training...')
                while not sess.should_stop():
                    self.train_step(sess, graph_data)

    def model(self, vocabulary_size=VOCABULARY_SIZE, embeddings_size=EMBEDDINGS_SIZE,
              output_classes=9):
        # global step
        self.global_step = training_util.get_or_create_global_step()

        # embeddings
        embeddings = self._load_embeddings(vocabulary_size, embeddings_size)

        # inputs
        self.inputs_text, self.expected_labels = self.dataset.read(self.batch_size)

        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            outputs = self.text_classification_model.model(self.inputs_text, output_classes,
                                                           embeddings=embeddings)
            self.prediction = outputs['prediction']

        self.saver = tf.train.Saver()
        # loss
        targets = self.text_classification_model.targets(self.expected_labels, output_classes)
        self.loss = self.text_classification_model.loss(targets, outputs)

        # metrics
        self.metrics = metrics.single_label(outputs['prediction'], tf.squeeze(targets),
                                            moving_average=False)

        return None

    def create_graph(self):
        return self.model()

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()
        self.saver.restore(session, tf.train.latest_checkpoint(self.log_dir))

    def begin(self):
        self.accumulated_loss = 0.0
        self.steps = 0

    def train_step(self, session, graph_data):
        loss, self.m = session.run([self.loss, self.metrics])
        self.steps += 1
        self.accumulated_loss += loss
        self.print_timestamp = time.time()
        elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
        m = 'samples: {} loss: {:0.4f} elapsed seconds: {} ' \
            'precision: {:0.3f} recall: {:0.3f} accuracy: {:0.3f}'
        print(m.format(self.steps * self.batch_size, self.accumulated_loss / self.steps,
                       elapsed_time, self.m['precision'], self.m['recall'], self.m['accuracy']))

    def end(self, session):
        print('steps: {}  dataset size: {}'.format(self.steps, self.steps * self.batch_size))
        print('precision: {}'.format(self.m['precision']))
        print('recall: {}'.format(self.m['recall']))
        print('accuracy: {}'.format(self.m['accuracy']))
        print('confusion matrix:')
        np.set_printoptions(precision=2, suppress=True, linewidth=120)
        print(self.m['confusion_matrix'])


class TextClassificationEvaluator(TextClassificationTest):
    """
    Helper class to run the training and create the model for the evaluation. See trainer.Trainer
    for more details.
    """

    def __init__(self, dataset, text_classification_model, logdir=DIR_TC_LOGDIR):
        super(TextClassificationEvaluator, self).__init__(dataset, text_classification_model, 1,
                                                          logdir)

    def model(self, vocabulary_size=VOCABULARY_SIZE, embeddings_size=EMBEDDINGS_SIZE,
              output_classes=9):
        # global step
        self.global_step = training_util.get_or_create_global_step()

        # embeddings
        embeddings = self._load_embeddings(vocabulary_size, embeddings_size)

        # inputs
        self.inputs_text, _ = self.dataset.read(self.batch_size)

        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            outputs = self.text_classification_model.model(self.inputs_text, output_classes,
                                                           embeddings=embeddings)
        self.prediction = outputs['prediction']

        self.saver = tf.train.Saver()

        return None

    def train_step(self, session, graph_data):
        predictions = session.run([self.prediction])[0][0]
        print('{},{}'.format(self.steps, ','.join([str(x) for x in predictions])))
        self.steps += 1

    def after_create_session(self, session, coord):
        super(TextClassificationEvaluator, self).after_create_session(session, coord)
        self.steps = 0
        print('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9')
