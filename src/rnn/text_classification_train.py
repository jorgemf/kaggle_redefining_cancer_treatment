import tensorflow as tf
import csv
import time
from datetime import timedelta
import sys
import numpy as np
from tensorflow.python.training import training_util
from tensorflow.contrib import slim
from tensorflow.python.ops import variables as tf_variables
from ..configuration import *
from .. import trainer, evaluator, metrics
from ..task_spec import get_task_spec
from .text_classification_dataset import TextClassificationDataset


def _load_embeddings(vocabulary_size, embeddings_size,
                     filename_prefix='embeddings', from_dir=DIR_DATA_WORD2VEC):
    embeddings = []
    embeddings_file = '{}_{}_{}'.format(filename_prefix, vocabulary_size, embeddings_size)
    with open(os.path.join(from_dir, embeddings_file), 'r') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            embeddings.append([float(r) for r in row])
    return embeddings


class TextClassificationTrainer(trainer.Trainer):
    """
    Helper class to run the training and create the model for the training. See trainer.Trainer for
    more details.
    """

    def __init__(self, dataset, text_classification_model, log_dir=DIR_TC_LOGDIR,
                 use_end_sequence=False, task_spec=None, max_steps=None):
        self.text_classification_model = text_classification_model
        self.use_end_sequence = use_end_sequence
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(TextClassificationTrainer, self).__init__(log_dir=log_dir, dataset=dataset,
                                                        task_spec=task_spec, max_steps=max_steps,
                                                        monitored_training_session_config=config)

    def model(self, input_text_begin, input_text_end, gene, variation, expected_labels, batch_size,
              vocabulary_size=VOCABULARY_SIZE, embeddings_size=EMBEDDINGS_SIZE, output_classes=9):
        # embeddings
        embeddings = _load_embeddings(vocabulary_size, embeddings_size)

        # global step
        self.global_step = training_util.get_or_create_global_step()

        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            outputs = self.text_classification_model.model(input_text_begin, input_text_end,
                                                           gene, variation, output_classes,
                                                           embeddings=embeddings,
                                                           batch_size=batch_size)

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
        self.metrics = metrics.single_label(outputs['prediction'], targets)

        # saver to save the model
        self.saver = tf.train.Saver()
        # check a nan value in the loss
        self.loss = tf.check_numerics(self.loss, 'loss is nan')

        return None

    def create_graph(self, dataset_tensor, batch_size):
        input_text_begin, input_text_end, gene, variation, expected_labels = dataset_tensor
        if not self.use_end_sequence:
            input_text_end = None
        return self.model(input_text_begin, input_text_end, gene, variation, expected_labels, batch_size)

    def step(self, session, graph_data):
        lr, _, loss, step, metrics = \
            session.run([self.learning_rate, self.optimizer, self.loss, self.global_step,
                         self.metrics])
        if self.is_chief and time.time() > self.print_timestamp + 5 * 60:
            self.print_timestamp = time.time()
            elapsed_time = str(timedelta(seconds=time.time() - self.init_time))
            m = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed seconds: {}  ' \
                'precision: {}  recall: {}  accuracy: {}'
            logging.info(m.format(step, loss, lr, elapsed_time,
                                  metrics['precision'], metrics['recall'], metrics['accuracy']))

    def after_create_session(self, session, coord):
        self.init_time = time.time()
        self.print_timestamp = time.time()


class TextClassificationTest(evaluator.Evaluator):
    """Evaluator for distributed training"""

    def __init__(self, dataset, text_classification_model, output_path, log_dir=DIR_TC_LOGDIR,
                 use_end_sequence=False,max_steps=None):
        self.use_end_sequence = use_end_sequence
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(TextClassificationTest, self).__init__(checkpoints_dir=log_dir, dataset=dataset,
                                                     output_path=output_path, max_steps=max_steps,
                                                     singular_monitored_session_config=config)
        self.text_classification_model = text_classification_model
        self.eval_writer = tf.summary.FileWriter(log_dir)

    def model(self, input_text_begin, input_text_end, gene, variation, expected_labels, batch_size,
              vocabulary_size=VOCABULARY_SIZE, embeddings_size=EMBEDDINGS_SIZE, output_classes=9):
        # embeddings
        embeddings = _load_embeddings(vocabulary_size, embeddings_size)
        # model
        with slim.arg_scope(self.text_classification_model.model_arg_scope()):
            outputs = self.text_classification_model.model(input_text_begin, input_text_end,
                                                           gene, variation, output_classes,
                                                           embeddings=embeddings,
                                                           batch_size=batch_size,
                                                           training=False)
        # loss
        targets = self.text_classification_model.targets(expected_labels, output_classes)
        loss = self.text_classification_model.loss(targets, outputs)
        self.accumulated_loss = tf.Variable(0.0, dtype=tf.float32, name='accumulated_loss',
                                            trainable=False)
        self.accumulated_loss = tf.assign_add(self.accumulated_loss, loss)
        step = tf.Variable(0, dtype=tf.int32, name='eval_step', trainable=False)
        step_increase = tf.assign_add(step, 1)
        self.loss = self.accumulated_loss / tf.cast(step_increase, dtype=tf.float32)
        tf.summary.scalar('loss', self.loss)
        # metrics
        self.metrics = metrics.single_label(outputs['prediction'], targets, moving_average=False)
        return None

    def create_graph(self, dataset_tensor, batch_size):
        input_text_begin, input_text_end, gene, variation, expected_labels = dataset_tensor
        if not self.use_end_sequence:
            input_text_end = None
        graph_data = self.model(input_text_begin, input_text_end, gene, variation,
                                expected_labels, batch_size)
        return graph_data

    def step(self, session, graph_data, summary_op):
        summary, self.loss_result, self.metrics_results = \
            session.run([summary_op, self.loss, self.metrics])
        return summary

    def end(self, session):
        super(TextClassificationTest, self).end(session)
        chk_step = int(self.lastest_checkpoint.split('-')[-1])
        m = 'step: {}  loss: {:0.4f}  precision: {}  recall: {}  accuracy: {}'
        logging.info(m.format(chk_step, self.loss_result, self.metrics_results['precision'],
                              self.metrics_results['recall'], self.metrics_results['accuracy']))

    def after_create_session(self, session, coord):
        # checkpoints_file = os.path.join(self.checkpoints_dir, 'checkpoint')
        # alt_checkpoints_dir = '{}_tp'.format(self.checkpoints_dir)
        # import glob
        # files = glob.glob('{}/model.ckpt-*.data-*'.format(alt_checkpoints_dir))
        # chk_step = 0
        # for f in files:
        #     num = f.split('model.ckpt-')[1].split('.')[0]
        #     num = int(num)
        #     if chk_step == 0 or num < chk_step:
        #         chk_step = num
        # if chk_step != 0:
        #     ckpt_files = glob.glob('{}/model.ckpt-{}.data-*'.format(alt_checkpoints_dir, chk_step))
        #     ckpt_files = [x.split('/')[-1] for x in ckpt_files]
        #     for f in ckpt_files + ['model.ckpt-{}.index', 'model.ckpt-{}.meta']:
        #         f = f.format(chk_step)
        #         os.rename(os.path.join(alt_checkpoints_dir, f), os.path.join(self.checkpoints_dir, f))
        #     with open(checkpoints_file, 'wb') as f:
        #         f.write('model_checkpoint_path: "./model.ckpt-{}"\n'.format(chk_step))
        #         f.write('all_model_checkpoint_paths: "./model.ckpt-{}"\n'.format(chk_step))
        super(TextClassificationTest, self).after_create_session(session, coord)
        # with open(checkpoints_file, 'wb') as f:
        #     f.write('model_checkpoint_path: "./model.ckpt-"\n')
        #     f.write('all_model_checkpoint_paths: "./model.ckpt-"\n')


class TextClassificationEval(evaluator.Evaluator):
    """Evaluator for text classification"""

    def __init__(self, dataset, text_classification_model, output_path, log_dir=DIR_TC_LOGDIR,
                 use_end_sequence=False):
        self.use_end_sequence = use_end_sequence
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(TextClassificationEval, self).__init__(checkpoints_dir=log_dir,
                                                     output_path=output_path,
                                                     infinite_loop=False,
                                                     singular_monitored_session_config=config)
        self.dataset = dataset
        self.text_classification_model = text_classification_model

    def model(self, input_text_begin, input_text_end, gene, variation, batch_size,
              vocabulary_size=VOCABULARY_SIZE, embeddings_size=EMBEDDINGS_SIZE, output_classes=9):
        # embeddings
        embeddings = _load_embeddings(vocabulary_size, embeddings_size)
        # global step
        self.global_step = training_util.get_or_create_global_step()
        self.global_step = tf.assign_add(self.global_step, 1)
        # model
        with tf.control_dependencies([self.global_step]):
            with slim.arg_scope(self.text_classification_model.model_arg_scope()):
                self.outputs = self.text_classification_model.model(input_text_begin, input_text_end,
                                                                    gene, variation, output_classes,
                                                                    embeddings=embeddings,
                                                                    batch_size=batch_size,
                                                                    training=False)
        # restore only the trainable variables
        self.saver = tf.train.Saver(var_list=tf_variables.trainable_variables())
        return self.outputs

    def create_graph(self, dataset_tensor, batch_size):
        input_text_begin, input_text_end, gene, variation = dataset_tensor
        if not self.use_end_sequence:
            input_text_end = None
        return self.model(input_text_begin, input_text_end, gene, variation, batch_size)

    def after_create_session(self, session, coord):
        super(TextClassificationEval, self).after_create_session(session, coord)
        print('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9')

    def step(self, session, graph_data, summary_op):
        step, predictions = session.run([self.global_step, self.outputs['prediction']])
        predictions = predictions[0]
        predictions = [p + 0.01 for p in predictions]  # penalize less the mistakes
        sum = np.sum(predictions)
        predictions = [p / sum for p in predictions]
        print('{},{}'.format(step, ','.join(['{:.3f}'.format(x) for x in predictions])))
        return None


import logging


def main(model, name, sentence_split=False, end_sequence=USE_END_SEQUENCE, batch_size=TC_BATCH_SIZE):
    """
    Main method to execute the text_classification models
    :param ModelSimple model: object model based on ModelSimple
    :param str name: name of the model
    :param bool sentence_split: whether to split the dataset in sentneces or not,
    only used for hatt model
    :param bool end_sequence: whether to use or not the end of the sequences in the models
    :param int batch_size: batch size of the models
    """
    logging.getLogger().setLevel(logging.INFO)
    log_dir = '{}_{}'.format(DIR_TC_LOGDIR, name)
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # execute the test with the train dataset
        dataset = TextClassificationDataset(type='train', sentence_split=sentence_split)
        tester = TextClassificationTest(dataset=dataset, text_classification_model=model,
                                        log_dir=log_dir,
                                        output_path=os.path.join(log_dir, 'test_trainset'),
                                        use_end_sequence=end_sequence)
        tester.run()
    elif len(sys.argv) > 1 and sys.argv[1] == 'validate':
        dataset = TextClassificationDataset(type='val', sentence_split=sentence_split)
        tester = TextClassificationTest(dataset=dataset, text_classification_model=model,
                                        log_dir=log_dir,
                                        output_path=os.path.join(log_dir, 'validate'),
                                        use_end_sequence=end_sequence)
        tester.run()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        # evaluate the data of the test dataset. We submit this output to kaggle
        dataset = TextClassificationDataset(type='test', sentence_split=sentence_split)
        evaluator = TextClassificationEval(dataset=dataset, text_classification_model=model,
                                           log_dir=log_dir,
                                           output_path=os.path.join(log_dir, 'test'),
                                           use_end_sequence=end_sequence)
        evaluator.run()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval_stage2':
        # evaluate the data of the test dataset. We submit this output to kaggle
        dataset = TextClassificationDataset(type='stage2_test', sentence_split=sentence_split)
        evaluator = TextClassificationEval(dataset=dataset, text_classification_model=model,
                                           log_dir=log_dir,
                                           output_path=os.path.join(log_dir, 'test_stage2'),
                                           use_end_sequence=end_sequence)
        evaluator.run()
    else:
        # training
        task_spec = get_task_spec(with_evaluator=USE_LAST_WORKER_FOR_VALIDATION)
        if task_spec.join_if_ps():
            # join if it is a parameters server and do nothing else
            return

        with(tf.gfile.Open(os.path.join(DIR_DATA_TEXT_CLASSIFICATION, 'train_set'))) as f:
            max_steps = int(TC_EPOCHS * len(f.readlines()) / batch_size)

        if task_spec.is_evaluator():
            dataset = TextClassificationDataset(type='val', sentence_split=sentence_split)
            # evaluator running in the last worker
            tester = TextClassificationTest(dataset=dataset, text_classification_model=model,
                                            log_dir=log_dir,
                                            output_path=os.path.join(log_dir, 'val'),
                                            use_end_sequence=end_sequence,
                                            max_steps=max_steps)
            tester.run()
        else:
            dataset = TextClassificationDataset(type='train', sentence_split=sentence_split)
            trainer = TextClassificationTrainer(dataset=dataset, text_classification_model=model,
                                                log_dir=log_dir, use_end_sequence=end_sequence,
                                                task_spec=task_spec, max_steps=max_steps)
            trainer.run(epochs=TC_EPOCHS, batch_size=batch_size)
