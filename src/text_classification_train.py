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
import text_classification_model


class MyTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self):
        dataset_spec = trainer.DatasetSpec('text_classification_train',
                                           DIR_DATA_TEXT_CLASSIFICATION,
                                           LOCAL_REPO, LOCAL_REPO_TC_PATH)
        super(MyTrainer, self).__init__(dataset_spec, DIR_W2V_LOGDIR)

    def model(self,
              batch_size=W2V_BATCH_SIZE,
              vocabulary_size=VOCABULARY_SIZE,
              embeddings_file=EMBEDDINGS_SIZE,
              output_classes=9,
              learning_rate_initial=TC_LEARNING_RATE_INITIAL,
              learning_rate_decay=TC_LEARNING_RATE_DECAY,
              learning_rate_decay_steps=TC_LEARNING_RATE_DECAY_STEPS):
        embeddings_file =
        self.global_step = training_util.get_or_create_global_step()

        # inputs
        self.inputs_text =  # TODO
        self.expected_labels =  # TODO

        # embeddings

        # model
        outputs = text_classification_model.model(self.inputs_text, output_classes)
        logits = outputs['logits']

        # loss
        targets = text_classification_model.targets(self.expected_labels, output_classes)
        self.loss = text_classification_model.loss(logits, targets)

        # learning rate & optimizer
        self.learning_rate = tf.train.exponential_decay(learning_rate_initial, self.global_step,
                                                        learning_rate_decay_steps,
                                                        learning_rate_decay,
                                                        staircase=True, name='learning_rate')
        sgd = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimizer = sgd.minimize(self.loss, global_step=self.global_step)

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

    def create_graph(self):
        return self.model()

    def train_step(self, session, graph_data):
        # lr, _, _, loss_val, step = session.run([self.learning_rate, self.training_op,
        #                                         self.loss, self.average_loss,
        #                                         self.global_step],
        #                                        feed_dict={
        #                                            self.inputs_text: text,
        #                                            self.expected_labels: labels,
        #                                        })
        # # if step % 100000 == 0:
        # if step % 100 == 0:
        #     elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
        #     m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}'
        #     print(m.format(step, loss_val, lr, elapsed_time))
        pass

    def after_create_session(self, session, coord):
        self.init_time = time.time()


if __name__ == '__main__':
    print('Loading dataset...')
    # TODO

    # start the training
    MyTrainer().train()
