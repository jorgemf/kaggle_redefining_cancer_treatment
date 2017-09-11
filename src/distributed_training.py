from task_spec import get_task_spec
from trainer import Trainer
from evaluator import Evaluator
import tensorflow as tf
from tensorflow.python.training import training_util


class DistributedTrainer(Trainer):
    """
    Example of class to run a distributed trainer with a dataset. The only tweak is to set
    the dataset to use shuffle and task_spec.
    Note the constructor receives a function to build the model, see
    model_fn_example(dataset_tensor, batch_size) for more information about this function.
    """

    def __init__(self, log_dir, dataset, model_fn, batch_size, epochs, task_spec, **kwargs):
        super(DistributedTrainer, self).__init__(log_dir=log_dir, task_spec=task_spec, **kwargs)
        self.dataset = dataset
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.epochs = epochs

    def create_graph(self):
        next_tensor = self.dataset.read(batch_size=self.batch_size,
                                        num_epochs=self.epochs,
                                        shuffle=True,
                                        task_spec=self.task_spec)
        return self.model_fn(dataset_tensor=next_tensor,
                             evaluation=False)

    def step(self, session, graph_data):
        session.run(graph_data)


class DistributedEvaluator(Evaluator):
    """
    Example of distributed evaluator. The only tweak is to set the dataset to use batch_size=1 and
    num_epochs=1. This way we guarantee the model see all the examples only once.
    Note the constructor receives a function to build the model, see
    model_fn_example(dataset_tensor, batch_size) for more information about this function.
    """

    def __init__(self, log_dir, dataset, model_fn, **kwargs):
        super(DistributedEvaluator, self).__init__(checkpoints_dir=log_dir, **kwargs)
        self.dataset = dataset
        self.model_fn = model_fn
        self.eval_writer = tf.summary.FileWriter(log_dir)

    def create_graph(self):
        next_tensor = self.dataset.read(batch_size=1, num_epochs=1)
        self.graph_data = self.model_fn(dataset_tensor=next_tensor,
                                        evaluation=True)
        self.global_step = training_util.get_global_step()
        self.summary_op = tf.summary.merge_all()
        return self.graph_data

    def after_create_session(self, session, coord):
        super(DistributedEvaluator, self).after_create_session(session, coord)
        self.summary_file = tf.summary.FileWriter(self.checkpoints_dir + '/eval')

    def end(self, session):
        super(DistributedEvaluator, self).end(session)
        # save summaries
        step = int(self.lastest_checkpoint.split('-')[-1])
        self.summary_file.add_summary(self.summary, step)

    def step(self, session, graph_data):
        self.summary = session.run(self.summary_op)


def model_fn_example(dataset_tensor, evaluation):
    """
    Example of the signature of the function that creates the model used in the
    Trainer and Evaluator.
    :param tf.Tensor dataset_tensor: the tensor created from the dataset
    :param bool evaluation: True if this model is for evaluation, False if it is for training.
    :return: returns the graph data, this is what session.run will execute during the training,
    for the test it will only execute the summary operator.
    """
    graph_data = None
    return graph_data


def launch_train_evaluation(model_fn, log_dir, epochs, train_batch_size, train_datasest,
                            test_dataset, **kwargs):
    """
    Launchs the training with an evaluator in the last worker. Only call this from distributed or it
    will fail.
    :param model_fn: function to create the model
    :param log_dir: directory for the logs/checkpoints
    :param epochs: number of epochs to perform the training
    :param train_batch_size: batch size of the trainer
    :param train_datasest: dataset for training
    :param test_dataset: dataset for evaluation
    :param kwargs: extra arguments for the trainer
    """
    task_spec = get_task_spec()
    if task_spec.num_workers <= 1:
        raise ValueError('More than one worker needed in order to perform a continuos evaluation')
    if task_spec.join_if_ps():
        return  # join if it is a parameters server and do nothing else
    if task_spec.index == task_spec.num_workers - 1:
        # run evaluator
        evaluator = DistributedEvaluator(log_dir=log_dir,
                                         dataset=test_dataset,
                                         model_fn=model_fn)
        evaluator.run()
    else:
        # run trainer (last worker is doing evaluation)
        task_spec.num_workers -= 1
        trainer = DistributedTrainer(log_dir=log_dir,
                                     dataset=train_datasest,
                                     model_fn=model_fn,
                                     batch_size=train_batch_size,
                                     epochs=epochs,
                                     task_spec=task_spec,
                                     **kwargs)
        trainer.run()
