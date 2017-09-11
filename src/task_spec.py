import os
import json
import argparse
import tensorflow as tf
import tensorport


class TaskSpec(object):
    """
    Specification for the task with the job name, index of the task, the parameter servers and
    the worker servers
    """

    def __init__(self, job_name='master', index=0, ps_hosts=None, worker_hosts=None):
        self.job_name = job_name
        self.index = index

        if ps_hosts and worker_hosts:
            ps = ps_hosts if isinstance(ps_hosts, list) else ps_hosts.split(',')
            worker = worker_hosts if isinstance(worker_hosts, list) else worker_hosts.split(',')
            self.cluster_spec = tf.train.ClusterSpec({'ps': ps, 'worker': worker, })
            self.num_workers = len(worker)
        else:
            self.cluster_spec = None
            self.num_workers = 1

    def is_chief(self):
        return self.index == 0

    def is_ps(self):
        return self.job_name == 'ps'

    def is_worker(self):
        return self.job_name == 'worker' or self.job_name == 'master'

    def join_if_ps(self):
        if self.is_ps():
            server = tf.train.Server(self.cluster_spec,
                                     job_name=self.job_name,
                                     task_index=self.index)
            server.join()
            return True
        return False


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
    args, _ = parser.parse_known_args()
    if args.job_name:
        return TaskSpec(job_name=args.job_name, index=args.task_index,
                        ps_hosts=args.ps_hosts, worker_hosts=args.worker_hosts)
    # get task from environment:
    if 'JOB_NAME' in os.environ:
        return TaskSpec(job_name=os.environ['JOB_NAME'], index=int(os.environ['TASK_INDEX']),
                        ps_hosts=os.environ.get('PS_HOSTS', None),
                        worker_hosts=os.environ.get('WORKER_HOSTS', None))
    if 'TF_CONFIG' in os.environ:
        env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = env.get('task', None) or {'type': 'master', 'index': 0}
        cluster_data = env.get('cluster', None) or {'ps': None, 'worker': None}
        return TaskSpec(job_name=task_data['type'], index=int(task_data['index']),
                        ps_hosts=cluster_data['ps'], worker_hosts=cluster_data['worker'])
    # return emtpy task spec for running in local
    return TaskSpec()


def get_logs_path(path):
    """
    Log dir specification, see: get_logs_path,
    https://tensorport.com/documentation/api/#get_logs_path
    :param str path: the path for the logs dir
    :return str: the real path for the logs
    """
    if path.startswith('gs://'):
        return path
    return tensorport.get_logs_path(path)


def get_data_path(dataset_name, local_root, local_repo='', path=''):
    """
    Dataset specification, see: get_data_path,
    https://tensorport.com/documentation/api/#get_data_path

    If local_root starts with gs:// we suppose a bucket in google cloud and return
    local_root / local_repo / local_path
    :param str name: TensorPort dataset repository name,
        e.g. user_name/repo_name
    :param str local_root: specifies the root directory for dataset.
          e.g. /home/username/datasets, gs://my-project/my_dir
    :param str local_repo: specifies the repo name inside the root data path.
          e.g. my_repo_data/
    :param str path: specifies the path inside the repository, (optional)
          e.g. train
    :return str: the real path of the dataset
    """
    if local_root.startswith('gs://'):
        return os.path.join(local_root, local_repo, path)
    return tensorport.get_data_path(
        dataset_name=dataset_name,
        local_root=local_root,
        local_repo=local_repo,
        path=path)
