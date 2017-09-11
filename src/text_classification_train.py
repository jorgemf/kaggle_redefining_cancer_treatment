import tensorflow as tf
import csv
import time
from datetime import timedelta
import sys
from tensorflow.python.training import training_util
from tensorflow.contrib import slim
from tensorflow.python.ops import variables as tf_variables
from src.configuration import *
import src.trainer as trainer
from src.task_spec import get_task_spec
import src.evaluator as evaluator
import src.metrics as metrics
from src.text_classification_dataset import TextClassificationDataset


def _load_embeddings(vocabulary_size, embeddings_size):
    embeddings = []
    embeddings_file = 'embeddings_{}_{}'.format(vocabulary_size, embeddings_size)
    with open(os.path.join(DIR_DATA_WORD2VEC, embeddings_file), 'r') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            embeddings.append([float(r) for r in row])
    return embeddings


class TextClassificationTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, text_classification_model, epochs=TC_EPOCHS,
                 batch_size=TC_BATCH_SIZE, log_dir=DIR_TC_LOGDIR, task_spec=None):
        self.dataset = dataset
        self.text_classification_model = text_classification_model
        self.batch_size = batch_size
        self.epochs = epochs
        super(TextClassificationTrainer, self).__init__(log_dir=log_dir, task_spec=task_spec)

    def model(self, input_texts, expected_labels, vocabulary_size=VOCABULARY_SIZE,
              embeddings_size=EMBEDDINGS_SIZE, output_classes=9):
        # embeddings
        embeddings = _load_embeddings(vocabulary_size, embeddings_size)

        # global step
        self.global_step = training_util.get_or_create_global_step()

        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            outputs = self.text_classification_model.model(input_texts, output_classes,
                                                           embeddings=embeddings)

        # loss
        targets = self.text_classification_model.targets(expected_labels, output_classes)
        self.loss = self.text_classification_model.loss(targets, outputs)
        tf.summary.scalar('loss', self.loss)

        # learning rate
        self.optimizer, self.learning_rate = \
            self.text_classification_model.optimize(self.loss, self.global_step)
        if self.learning_rate is not None:
            tf.summary.scalar('learning_rate', self.learning_rate)

        # metrics
        self.metrics = metrics.single_label(outputs['prediction'], tf.squeeze(targets))

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
        input_texts, expected_labels = next_tensor
        return self.model(input_texts, expected_labels)

    def step(self, session, graph_data):
        lr, _, loss, step, metrics = \
            session.run([self.learning_rate, self.optimizer, self.loss, self.global_step,
                         self.metrics])
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


class TextClassificationTest(evaluator.Evaluator):
    """Evaluator for distributed training"""

    def __init__(self, dataset, text_classification_model, log_dir=DIR_TC_LOGDIR):
        super(TextClassificationTest, self).__init__(checkpoints_dir=log_dir)
        self.dataset = dataset
        self.text_classification_model = text_classification_model
        self.eval_writer = tf.summary.FileWriter(log_dir)

    def model(self, input_texts, expected_labels, vocabulary_size=VOCABULARY_SIZE,
              embeddings_size=EMBEDDINGS_SIZE, output_classes=9):
        # embeddings
        embeddings = _load_embeddings(vocabulary_size, embeddings_size)
        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            outputs = self.text_classification_model.model(input_texts, output_classes,
                                                           embeddings=embeddings,
                                                           training=False)
        # loss
        targets = self.text_classification_model.targets(expected_labels, output_classes)
        self.loss = self.text_classification_model.loss(targets, outputs)
        tf.summary.scalar('loss', self.loss)
        # metrics
        self.metrics = metrics.single_label(outputs['prediction'], tf.squeeze(targets),
                                            moving_average=False)
        return None

    def create_graph(self):
        next_tensor = self.dataset.read(batch_size=1, num_epochs=1)
        input_texts, expected_labels = next_tensor
        self.graph_data = self.model(input_texts, expected_labels)
        self.summary_op = tf.summary.merge_all()
        return self.graph_data

    def after_create_session(self, session, coord):
        super(TextClassificationTest, self).after_create_session(session, coord)
        self.summary_file = tf.summary.FileWriter(self.checkpoints_dir + '/test')

    def end(self, session):
        super(TextClassificationTest, self).end(session)
        # save summaries
        step = int(self.lastest_checkpoint.split('-')[-1])
        self.summary_file.add_summary(self.summary, step)

    def step(self, session, graph_data):
        self.summary = session.run(self.summary_op)


class TextClassificationEval(evaluator.Evaluator):
    """Evaluator for text classification"""

    def __init__(self, dataset, text_classification_model, log_dir=DIR_TC_LOGDIR):
        super(TextClassificationEval, self).__init__(checkpoints_dir=log_dir)
        self.dataset = dataset
        self.text_classification_model = text_classification_model

    def model(self, input_texts, vocabulary_size=VOCABULARY_SIZE,
              embeddings_size=EMBEDDINGS_SIZE, output_classes=9):
        # global step
        self.global_step = training_util.get_or_create_global_step()
        # embeddings
        embeddings = _load_embeddings(vocabulary_size, embeddings_size)
        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            self.outputs = self.text_classification_model.model(input_texts, output_classes,
                                                                embeddings=embeddings,
                                                                training=False)
        # restore only the trainable variables
        self.saver = tf.train.Saver(var_list=tf_variables.trainable_variables())
        return None

    def create_graph(self):
        input_texts = self.dataset.read(batch_size=1, num_epochs=1)
        self.graph_data = self.model(input_texts)

    def after_create_session(self, session, coord):
        super(TextClassificationEval, self).after_create_session(session, coord)
        self.saver.restore(session, tf.train.latest_checkpoint(self.log_dir))
        print('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9')

    def step(self, session, graph_data):
        step, predictions = session.run([self.global_step, self.outputs])
        print('{},{}'.format(step - 1, ','.join([str(x) for x in predictions[0]])))


def main(model, name, sentence_split=False):
    """
    Main method to execute the text_classification models
    :param ModelSimple model: object model based on ModelSimple
    :param str name: name of the model
    :param bool sentence_split: whether to split the dataset in sentneces or not,
    only used for hatt model
    """
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # execute the test with the train dataset
        dataset = TextClassificationDataset(type='train', sentence_split=sentence_split)
        tester = TextClassificationTest(dataset=dataset, text_classification_model=model,
                                        log_dir='{}_{}'.format(DIR_TC_LOGDIR, name))
        tester.run()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # evaluate the data of the test dataset. We submit this output to kaggle
        dataset = TextClassificationDataset(type='test', sentence_split=sentence_split)
        evaluator = TextClassificationTest(dataset=dataset, text_classification_model=model,
                                           log_dir='{}_{}'.format(DIR_TC_LOGDIR, name))
        evaluator.run()
    else:
        # training
        task_spec = get_task_spec()
        if task_spec.join_if_ps():
            # join if it is a parameters server and do nothing else
            return

        # use the train dataset for training and testing. Not what ideally we would do in most cases
        dataset = TextClassificationDataset(type='train', sentence_split=sentence_split)
        if task_spec.num_workers <= 1:
            # single machine training, we don't run the evaluator
            trainer = TextClassificationTrainer(dataset=dataset, text_classification_model=model,
                                                log_dir='{}_{}'.format(DIR_TC_LOGDIR, name))
            trainer.run()
        elif task_spec.index == task_spec.num_workers - 1:
            # evaluator running in the last worker
            tester = TextClassificationTest(dataset=dataset, text_classification_model=model,
                                            log_dir='{}_{}'.format(DIR_TC_LOGDIR, name))
            tester.run()
        else:
            # run trainer in the rest of the workers
            task_spec.num_workers -= 1
            trainer = TextClassificationTrainer(dataset=dataset, text_classification_model=model,
                                                log_dir='{}_{}'.format(DIR_TC_LOGDIR, name),
                                                task_spec=task_spec)
            trainer.run()
