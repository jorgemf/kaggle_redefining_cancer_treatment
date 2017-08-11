import logging
import os
import json
import argparse
import time

from tensorport import get_data_path, get_logs_path
import tensorflow as tf
from tensorflow.python.training import session_run_hook


class TaskSpec(object):
    """
    Specification for the task with the job name, index of the task, the parameter servers and
    the worker servers
    """

    def __init__(self, job_name='mater', index=0, ps_hosts=None, worker_hosts=None):
        self.job_name = job_name
        self.index = index

        if ps_hosts and worker_hosts:
            ps = ps_hosts if isinstance(ps_hosts, list) else ps_hosts.split(',')
            worker = worker_hosts if isinstance(worker_hosts, list) else worker_hosts.split(',')
            self.cluster_spec = tf.train.ClusterSpec({'ps': ps, 'worker': worker, })
        else:
            self.cluster_spec = None


def get_task_spec():
    """
    Loads the task information from the command line or the enviorment variables (if the command
    line parameters are not set) and returns a TaskSpec object
    :return TaskSpec: a TaskSpec object with the information about the task
    """
    # get task from parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', dest='job_name', default=None)
    parser.add_argument('--task_index', dest='task_index', default=None)
    parser.add_argument('--ps_hosts', dest='ps_hosts', default=None)
    parser.add_argument('--worker_hosts', dest='worker_hosts', default=None)
    args = parser.parse_args()
    if args.job_name:
        return TaskSpec(job_name=args.job_name, index=args.task_index,
                        ps_hosts=args.ps_hosts, worker_hosts=args.worker_hosts)
    # get task from environment:
    if 'JOB_NAME' in os.environ:
        return TaskSpec(job_name=os.environ['JOB_NAME'], index=os.environ['TASK_INDEX'],
                        ps_hosts=os.environ.get(['PS_HOSTS'], None),
                        worker_hosts=os.environ.get(['WORKER_HOSTS'], None))
    if 'TF_CONFIG' in os.environ:
        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = env.get('task', None) or {'type': 'master', 'index': 0}
        cluster_data = env.get('cluster', None) or {'ps': None, 'worker': None}
        return TaskSpec(job_name=task_data['type'], index=task_data['index'],
                        ps_hosts=cluster_data['ps'], worker_hosts=cluster_data['worker'])
    # return emtpy task spec for running in local
    return TaskSpec()


class DatasetSpec(object):
    """
    Dataset specification, see: get_data_path,
    https://tensorport.com/documentation/api/#get_data_path
    """

    def __init__(self, dataset_name, local_root, local_repo, path=''):
        """
        :param string name: TensorPort dataset repository name,
            e.g. tensorport/mnist
        :param string local_root: specifies the root directory for dataset.
              e.g. /home/username/datasets
        :param stromg local_repo: specifies the repo name inside the root data path.
              e.g. mnist
        :param string path: specifies the path inside the repository, (optional)
              e.g. train
        """
        self.name = dataset_name
        self.local_root = local_root
        self.local_repo = local_repo
        self.path = path


class Trainer(session_run_hook.SessionRunHook):
    """
    Default class for training. This class extends from SessionRunHook and it is added to the hooks
    in the MonitoredTrainingSession
    """

    def __init__(self, dataset_spec, log_dir, max_time=-1,
                 save_checkpoint_secs=600, save_summaries_steps=100, log_step_count_steps=100,
                 monitored_training_session_config=None):
        """
        :param DatasetSpec dataset_spec: dataset specification
        :param str log_dir: directory where logs are stored
        :param int max_time: max time to run the training, by default is -1 to run indefinitely
        :param int save_checkpoint_secs: seconds to save the checkpoints
        :param int save_summaries_steps: steps to save the summaries
        :param int log_step_count_steps: steps to log the steps_count
        :param tf.ConfigProto monitored_training_session_config: an instance of tf.ConfigProto,
        the configuration for the monitored training session
        Activate these hooks if is_chief==True`, ignore otherwise.
        """
        self.data_dir = get_data_path(
            dataset_name=dataset_spec.name,
            local_root=dataset_spec.local_root,
            local_repo=dataset_spec.local_repo,
            path=dataset_spec.path
        )
        self.log_dir = get_logs_path(root=log_dir)
        self.save_checkpoint_secs = save_checkpoint_secs
        self.save_summaries_steps = save_summaries_steps
        self.log_step_count_steps = log_step_count_steps
        self.monitored_training_session_config = monitored_training_session_config
        self.max_time = max_time
        logging.info('Log dir: {}', self.log_dir)
        logging.info('Data dir: {}', self.data_dir)

    def train(self):
        """
        Starts the training
        """
        task_spec = get_task_spec()
        if task_spec.cluster_spec:
            server = tf.train.Server(task_spec.cluster_spec,
                                     job_name=task_spec.job_name,
                                     task_index=task_spec.index)
            if task_spec.job_name == 'ps':
                server.join()
            target = server.target
            device = tf.train.replica_device_setter(
                worker_device='/job:worker/task:{}'.format(task_spec.index),
                cluster=task_spec.cluster_spec)
        else:
            device = None
            target = ''

        logging.info('Creating graph...')
        with tf.Graph().as_default():
            with tf.device(device):
                graph_data = self.create_graph()

            hooks, chief_only_hooks = self.create_hooks(graph_data)
            if hooks is None:
                hooks = []
            hooks.append(self)
            if self.max_time > 0:
                hooks.append(StopAtTimeHook(self.max_time))

            logging.info('Creating MonitoredTrainingSession...')
            with tf.train.MonitoredTrainingSession(
                    master=target,
                    is_chief=(task_spec.index == 0),
                    checkpoint_dir=self.log_dir,
                    save_checkpoint_secs=self.save_checkpoint_secs,
                    save_summaries_steps=self.save_summaries_steps,
                    log_step_count_steps=self.log_step_count_steps,
                    config=self.monitored_training_session_config,
                    hooks=hooks,
                    chief_only_hooks=chief_only_hooks) as sess:

                logging.info('Starting training...')

                while not sess.should_stop():
                    self.train_step(sess, graph_data)

    def create_graph(self):
        """
        Function to create the graph
        :return: Information related with the graph. It will be passed to other functions
        """
        raise NotImplementedError('Should have implemented this')

    def create_hooks(self, graph_data):
        """
        Creates the hooks for the session. This function is called after the graph is created and
        before the session is created.
        :param graph_data: the graph data returned create_graph
        :return: A tuple with two lists of hooks or none. First list if the hooks for all nodes and
        the second list are the hooks only for the master node.
        """
        return [], []

    def train_step(self, session, graph_data):
        """
        Function to run one time step.
        :param session: the session
        :param graph_data: the graph data returned in create_graph
        """
        raise NotImplementedError('Should have implemented this')


class StopAtTimeHook(session_run_hook.SessionRunHook):
    """Hook that requests stop after a specified time."""

    def __init__(self, time_running):
        """
        :param int time_running: Maximum time running
        """
        self._time_running = time_running

    def begin(self):
        self._end_time = time.time() + self._time_running

    def after_run(self, run_context, run_values):
        if time.time() > self._end_time:
            run_context.request_stop()
