import tensorflow as tf
from configuration import *


def model(input_text, output_classes, embeddings, num_hidden=TC_MODEL_HIDDEN,
          num_layers=TC_MODEL_LAYERS, dropout=TC_MODEL_DROPOUT):
    """
    Creates a model for text classification
    :param tf.Tensor input_text: the input data, the text as
    [batch_size, text_vector_max_length, embeddings_size]
    :param int output_classes: the number of output classes for the classifier
    :param List[List[float]] embeddings: a matrix with the embeddings for the embedding lookup
    :param int num_hidden: number of hidden GRU cells in every layer
    :param int num_layers: numer of layers of the model
    :param float dropout: dropout value between layers
    :return Dict[str,tf.Tensor]: a dict with logits and prediction tensors
    """
    embeddings = tf.Variable(embeddings, name='embeddings')
    embedded_sequence = tf.nn.embedding_lookup(embeddings, input_text)

    # Recurrent network.
    network = tf.contrib.rnn.GRUCell(num_hidden)
    network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=dropout)
    network = tf.contrib.rnn.MultiRNNCell([network] * num_layers)
    output, _ = tf.nn.dynamic_rnn(network, embedded_sequence, dtype=tf.float32)
    # Select last output.
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    # full connected + softmax
    logits = tf.contrib.layers.fully_connected(last, output_classes)
    prediction = tf.nn.softmax(logits)
    return {'logits': logits, 'prediction': prediction}


def targets(labels, output_classes):
    """
    Transfor a vector of labels into a matrix of one hot encoding labels
    :param tf.Tensor labels: an array of labels with dimension [batch_size]
    :param int output_classes: the total number of output classes
    :return tf.Tensor: a tensorflow tensor
    """
    targets = tf.one_hot(labels, axis=-1, depth=output_classes, on_value=1.0, off_value=0.0)
    return targets


def loss(logits, targets):
    """
    Calculates the softmax cross entropy loss
    :param tf.Tensor logits: logits output of the model
    :param tf.Tensor targets: targets with the one hot encoding labels
    :return tf.Tensor : a tensor with the loss value
    """
    loss = tf.losses.softmax_cross_entropy(targets, logits)
    loss = tf.reduce_mean(loss)
    return loss


def length(sequence):
    """
    Calculate the length of a sequence padded with -1 at the end of the sequence.
    :param tf.Tensor sequence: tensor with the sequence
    :return tf.Tensor: the length of the sequence as a tensor
    """
    used = tf.sign(sequence, 2)
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length, sequence
