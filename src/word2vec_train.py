import process_data
from random import shuffle
import tensorflow as tf
import math
import numpy as np
import random


def generate_training_samples(text_lines, samples_augmentation_per_word, window_adjacent_words,
                              close_words_size, window_close_words, word_frequency_dict):
    data = []
    for text_line in text_lines:
        for i, word in enumerate(text_line):
            aw_min = max(0, i - window_adjacent_words)
            aw_max = min(len(text_line), i + window_adjacent_words + 1)
            adjacent_words = text_line[aw_min:i] + text_line[i + 1:aw_max]
            nsw_min = max(0, min(aw_min, i - window_close_words))
            nsw_max = min(len(text_line), max(aw_max, i + window_close_words + 1))
            for _ in range(samples_augmentation_per_word):
                negative_samples_words_list = text_line[nsw_min:aw_min] + text_line[aw_max:nsw_max]
                negative_samples_words = select_random_labels(negative_samples_words_list,
                                                              close_words_size, word_frequency_dict)
                context = adjacent_words + negative_samples_words
                for label in context:
                    data.append([word, label])
                if len(negative_samples_words) == 0:
                    break
    return data


def select_random_labels(labels, num_labels, word_frequency_dict):
    samples = []
    max_labels = min(num_labels, len(labels))
    for i in range(max_labels):
        shuffle(labels)
        probabilities = [word_frequency_dict[word] for word in labels]
        probabilities_sum = np.sum(probabilities)
        probabilities = [p / probabilities_sum for p in probabilities]
        r = random.random()
        p_sum = 0.0
        i = 0
        while p_sum < r:
            p_sum += probabilities[i]
            i += 1
        i -= 1
        word = labels[i]
        del labels[i]
        samples.append(word)
    return samples


def generate_batch(data, batch_size, epochs):
    batch = []
    len_data = len(data)
    for epoch in range(epochs):
        shuffle(data)
        for i, sample in enumerate(data):
            batch.append(sample)
            if len(batch) == batch_size:
                inputs, outputs = zip(*batch)
                yield inputs, outputs, epoch, float(i) / len_data
                batch = []


def load_word2vec_data(filename, vocabulary_size=30000):
    filename = '{}_{}'.format(filename, vocabulary_size)
    filename_dict = '{}_dict'.format(filename)
    symbols_dict, encoded_text = process_data.load_word2vec_data(filename, filename_dict)
    filename_count = '{}_count'.format(filename)
    total_count = 0
    with open(filename_count, 'r') as f:
        word_frequency_dict = {}
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                data = line.split(' = ')
                symbol = symbols_dict[data[0].strip()]
                count = int(data[1].strip())
                if symbol in symbols_dict:
                    word_frequency_dict[symbol] += count
                else:
                    word_frequency_dict[symbol] = count
                total_count += count
    for key in word_frequency_dict.keys():
        word_frequency_dict[key] = float(word_frequency_dict[key]) / total_count

    return symbols_dict, encoded_text, word_frequency_dict


def model(batch_size, vocabulary_size, embedding_size, num_negative_samples):
    graph = tf.Graph()
    with graph.as_default():
        input_word = tf.placeholder(tf.int32, shape=[batch_size])
        output_label = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            matrix_dimension = [vocabulary_size, embedding_size]
            embeddings = tf.Variable(tf.random_uniform(matrix_dimension, -1.0, 1.0))

            embed = tf.nn.embedding_lookup(embeddings, input_word)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal(matrix_dimension, stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=output_label,
                                  inputs=embed, num_sampled=num_negative_samples,
                                  num_classes=vocabulary_size, partition_strategy='div')
        loss = tf.reduce_mean(nce_loss)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        # initializing function
        init = tf.global_variables_initializer()
        # summaries and moving average
        ema = tf.train.ExponentialMovingAverage(0.8)
        ema.apply([loss])
        average_loss = ema.average(loss)
        tf.summary.scalar('loss', loss)

    return graph, input_word, output_label, init, loss, average_loss, optimizer, embeddings


if __name__ == '__main__':
    epochs = 10  # iterations over the whole dataset where
    # (dataset size = len(dataset) * samples_augmentation_per_word)
    batch_size = 128  # batch size for the training
    embedding_size = 128  # embedding size
    window_adjacent_words = 1  # adjacent words to be added to the context
    close_words_size = 2  # close words (non-adjacent) to be added to the context
    window_close_words = 6  # maximum distance between the target word and the close words
    samples_augmentation_per_word = 3  # number of context generated per target word
    negative_num_samples = 64  # Number of negative examples to sample.

    print('Loading dataset...')
    symbols_dict, encoded_text, word_frequency_dict = load_word2vec_data(
        'data/generated/word2vec_dataset')
    vocabulary_size = len(set(symbols_dict.values()))
    print('Generating data samples for the word2vec model...')
    data = generate_training_samples(encoded_text,
                                     samples_augmentation_per_word,
                                     window_adjacent_words,
                                     close_words_size, window_close_words,
                                     word_frequency_dict)
    print('Creating graph...')
    graph, input_word, output_label, init, loss, average_loss, optimizer, embeddings = \
        model(batch_size, vocabulary_size, embedding_size, negative_num_samples)

    print('Training...')
    with tf.Session(graph=graph) as session:
        init.run()
        average_loss = 0

        for words, labels, epoch, percentage_epoch in generate_batch(data, batch_size, epochs):
            _, _, loss_val = session.run([optimizer, loss, average_loss],
                                         feed_dict={input_word: words, output_label: labels})
            epoch = epoch + percentage_epoch
            print('epoch: {0.2f} loss: {:0.4f}'.format(epoch, loss_val))

    embeddings_eval = embeddings.eval()
    norm = np.sqrt(np.sum(np.square(embeddings_eval)))
    normalized_embeddings = embeddings_eval / norm
    print('Saving embeddings...')
