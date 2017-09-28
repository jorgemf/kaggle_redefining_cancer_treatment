import tensorflow as tf
from tensorflow.python.training import training_util
from . import evaluator, metrics
from .configuration import *
from .doc2vec_train_doc_prediction import doc2vec_prediction_model


class DocPredictionEval(evaluator.Evaluator):
    def __init__(self, dataset, log_dir=DIR_D2V_DOC_LOGDIR):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        super(DocPredictionEval, self).__init__(checkpoints_dir=log_dir,
                                                output_path=os.path.join(log_dir,
                                                                         dataset.type),
                                                dataset=dataset,
                                                singular_monitored_session_config=config,
                                                infinite_loop=False)

    def model(self, input_vectors, output_label, batch_size, embedding_size=EMBEDDINGS_SIZE,
              output_classes=9):
        self.global_step = training_util.get_or_create_global_step()

        logits, targets = doc2vec_prediction_model(input_vectors, output_label, batch_size,
                                                   is_training=False, embedding_size=embedding_size,
                                                   output_classes=output_classes)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        global_step_increase = tf.assign_add(self.global_step, 1)
        self.accumulated_loss = tf.Variable(0.0, dtype=tf.float32, name='accumulated_loss', trainable=False)
        self.accumulated_loss = tf.assign_add(self.accumulated_loss, tf.reduce_sum(loss))
        with tf.control_dependencies([global_step_increase, self.accumulated_loss]):
            self.prediction = tf.nn.softmax(logits)
            self.metrics = metrics.single_label(self.prediction, targets, moving_average=False)
        tf.summary.scalar('loss', self.accumulated_loss / (global_step_increase * batch_size))
        # saver to save the model
        self.saver = tf.train.Saver()

        return None

    def create_graph(self, dataset_tensor, batch_size):
        input_vectors, output_label = dataset_tensor
        self.batch_size = batch_size
        return self.model(input_vectors, output_label, batch_size)

    def step(self, session, graph_data, summary_op):
        self.num_steps, self.final_metrics, self.final_loss, summary = \
            session.run([self.global_step, self.metrics, self.accumulated_loss, summary_op])
        return summary

    def end(self, session):
        cm = self.final_metrics['confusion_matrix']
        data_size = self.num_steps * self.batch_size
        loss = self.final_loss / data_size
        print('Loss: {}'.format(loss))
        print('Confusion matrix:')
        for r in cm:
            print('\t'.join([str(x) for x in r]))


class DocPredictionInference(evaluator.Evaluator):
    def __init__(self, dataset):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        log_dir = DIR_D2V_DOC_LOGDIR
        super(DocPredictionInference, self).__init__(checkpoints_dir=log_dir,
                                                     output_path=os.path.join(log_dir,
                                                                              dataset.type),
                                                     dataset=dataset,
                                                     singular_monitored_session_config=config,
                                                     infinite_loop=False)

    def model(self, input_vectors, batch_size, embedding_size=EMBEDDINGS_SIZE, output_classes=9):
        self.global_step = training_util.get_or_create_global_step()

        logits, _ = doc2vec_prediction_model(input_vectors, None, batch_size,
                                             is_training=False, embedding_size=embedding_size,
                                             output_classes=output_classes)
        global_step_increase = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            self.prediction = tf.nn.softmax(logits)
        # saver to save the model
        self.saver = tf.train.Saver()

        return self.prediction

    def create_graph(self, dataset_tensor, batch_size):
        return self.model(dataset_tensor, batch_size)

    def after_create_session(self, session, coord):
        super(DocPredictionInference, self).after_create_session(session, coord)
        print('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9')

    def step(self, session, graph_data, summary_op):
        step, predictions = session.run([self.global_step, self.prediction])
        print('{},{}'.format(step, ','.join([str(x) for x in predictions[0]])))
        return None
