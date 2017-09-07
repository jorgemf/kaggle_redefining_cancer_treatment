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
from tensorflow.python.ops import variables as tf_variables
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
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with SingularMonitoredSession(hooks=hooks, config=config) as sess:
                logging.info('Starting training...')
                while not sess.should_stop():
                    self.train_step(sess, graph_data)

    def model(self, vocabulary_size=VOCABULARY_SIZE, embeddings_size=EMBEDDINGS_SIZE,
              output_classes=9, use_metrics=True, use_loss=True):
        # global step
        self.global_step = training_util.get_or_create_global_step()
        global_step_increase = tf.assign_add(self.global_step, 1)

        # embeddings
        embeddings = self._load_embeddings(vocabulary_size, embeddings_size)

        # inputs
        self.inputs_text, self.expected_labels = self.dataset.read(self.batch_size, epochs=1)

        # model
        with tf.control_dependencies([global_step_increase]):
            with slim.arg_scope(self.text_classification_model.model_arg_scope()):
                outputs = self.text_classification_model.model(self.inputs_text, output_classes,
                                                               embeddings=embeddings,
                                                               training=False)
                self.prediction = outputs['prediction']

        # restore only the trainable variables
        self.saver = tf.train.Saver(var_list=tf_variables.trainable_variables())

        if use_loss or use_metrics:
            # loss
            targets = self.text_classification_model.targets(self.expected_labels, output_classes)
            self.loss = self.text_classification_model.loss(targets, outputs)

        if use_metrics:
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
        step, loss, self.m = session.run([self.global_step, self.loss, self.metrics])
        self.accumulated_loss += loss
        self.print_timestamp = time.time()
        elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
        m = 'samples: {} loss: {:0.4f} elapsed (h:m:s): {} ' \
            'precision: {:0.3f} recall: {:0.3f} accuracy: {:0.3f}'
        print(m.format(step * self.batch_size, self.accumulated_loss / step,
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

    def create_graph(self):
        return self.model(use_metrics=False, use_loss=False)

    def train_step(self, session, graph_data):
        step, predictions = session.run([self.global_step, self.prediction])
        print('{},{}'.format(step - 1, ','.join([str(x) for x in predictions[0]])))

    def after_create_session(self, session, coord):
        super(TextClassificationEvaluator, self).after_create_session(session, coord)
        print('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9')

    def end(self, session):
        pass
