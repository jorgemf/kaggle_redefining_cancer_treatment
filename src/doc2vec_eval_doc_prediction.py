import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util
from . import evaluator, metrics
from .configuration import *
from .doc2vec_train_doc_prediction import doc2vec_prediction_model
from .doc2vec_train_doc_prediction import DocPredictionDataset


class DocPredictionEval(evaluator.Evaluator):
    def __init__(self, dataset, log_dir=DIR_D2V_DOC_LOGDIR):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(DocPredictionEval, self).__init__(checkpoints_dir=log_dir,
                                                output_path=os.path.join(log_dir,
                                                                         dataset.type),
                                                dataset=dataset,
                                                singular_monitored_session_config=config,
                                                infinite_loop=True)
        self.best_loss = -1

    def model(self, input_vectors, input_gene, input_variation, output_label, batch_size,
              embedding_size=EMBEDDINGS_SIZE,
              output_classes=9):
        logits, targets = doc2vec_prediction_model(input_vectors, input_gene, input_variation,
                                                   output_label, batch_size,
                                                   is_training=False, embedding_size=embedding_size,
                                                   output_classes=output_classes)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        self.global_step = training_util.get_or_create_global_step()
        global_step_increase = tf.assign_add(self.global_step, 1)
        self.accumulated_loss = tf.Variable(0.0, dtype=tf.float32, name='accumulated_loss',
                                            trainable=False)
        self.accumulated_loss = tf.assign_add(self.accumulated_loss, tf.reduce_sum(loss))
        self.prediction = tf.nn.softmax(logits)
        self.metrics = metrics.single_label(self.prediction, targets, moving_average=False)
        steps = tf.cast(global_step_increase, dtype=tf.float32)
        tf.summary.scalar('loss', self.accumulated_loss / (steps * batch_size))
        return None

    def create_graph(self, dataset_tensor, batch_size):
        input_vectors, input_gene, input_variation, output_label = dataset_tensor
        self.batch_size = batch_size
        return self.model(input_vectors, input_gene, input_variation, output_label, batch_size)

    def step(self, session, graph_data, summary_op):
        self.num_steps, self.final_metrics, self.final_loss, summary = \
            session.run([self.global_step, self.metrics, self.accumulated_loss, summary_op])
        return summary

    def after_create_session(self, session, coord):
        super(DocPredictionEval, self).after_create_session(session, coord)

    def end(self, session):
        super(DocPredictionEval, self).end(session)
        cm = self.final_metrics['confusion_matrix']
        data_size = self.num_steps * self.batch_size
        loss = self.final_loss / data_size
        print('Loss: {}'.format(loss))
        print('Confusion matrix:')
        for r in cm:
            print('\t'.join([str(x) for x in r]))
        if self.best_loss < 0 or loss < self.best_loss:
            self.best_loss = loss
            self.copy_checkpoint_as_best()


class DocPredictionInference(evaluator.Evaluator):
    def __init__(self, dataset, log_dir=DIR_D2V_DOC_LOGDIR):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(DocPredictionInference, self).__init__(checkpoints_dir=log_dir,
                                                     output_path=os.path.join(log_dir,
                                                                              dataset.type),
                                                     dataset=dataset,
                                                     singular_monitored_session_config=config,
                                                     infinite_loop=False)

    def model(self, input_vectors, input_gene, input_variation, batch_size,
              embedding_size=EMBEDDINGS_SIZE, output_classes=9):
        self.global_step = training_util.get_or_create_global_step()

        logits, _ = doc2vec_prediction_model(input_vectors, input_gene, input_variation,
                                             None, batch_size,
                                             is_training=False, embedding_size=embedding_size,
                                             output_classes=output_classes)
        global_step_increase = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            self.prediction = tf.nn.softmax(logits)
        return self.prediction

    def end(self, session):
        pass

    def create_graph(self, dataset_tensor, batch_size):
        input_vectors, input_gene, input_variation, _ = dataset_tensor
        return self.model(input_vectors, input_gene, input_variation, batch_size)

    def after_create_session(self, session, coord):
        super(DocPredictionInference, self).after_create_session(session, coord)
        print('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9')

    def step(self, session, graph_data, summary_op):
        step, predictions = session.run([self.global_step, self.prediction])
        predictions = predictions[0]
        predictions = [p + 0.01 for p in predictions]  # penalize less the mistakes
        sum = np.sum(predictions)
        predictions = [p / sum for p in predictions]
        print('{},{}'.format(step, ','.join(['{:.3f}'.format(x) for x in predictions])))
        return None


if __name__ == '__main__':
    import logging

    logging.getLogger().setLevel(logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == 'val':
        # get validation error
        evaluator = DocPredictionEval(dataset=DocPredictionDataset(type='val'),
                                      log_dir=os.path.join(DIR_D2V_DOC_LOGDIR))
        evaluator.run()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        # get validation error
        evaluator = DocPredictionInference(dataset=DocPredictionDataset(type='stage2_test'),
                                           log_dir=os.path.join(DIR_D2V_DOC_LOGDIR, 'val'))
        evaluator.run()
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        # get validation error
        evaluator = DocPredictionEval(dataset=DocPredictionDataset(type='train'),
                                      log_dir=os.path.join(DIR_D2V_DOC_LOGDIR))
        evaluator.run()
