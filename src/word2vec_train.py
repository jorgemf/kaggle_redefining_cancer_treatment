import process_data
import tensorflow as tf
import math
import numpy as np
import random
import csv
import time
from datetime import timedelta


def generate_training_samples(text_lines, epochs, window_adjacent_words, close_words_size,
                              window_close_words, word_frequency_dict):
    probabilities_dict = {}
    for k, v in word_frequency_dict.items():
        probabilities_dict[k] = -math.log(v)
    for _ in range(epochs):
        random.shuffle(text_lines)
        for text_line in text_lines:
            probabilities_tl = [probabilities_dict[w] for w in text_line]
            len_text_line = len(text_line)
            for i, word in enumerate(text_line):
                aw_min = max(0, i - window_adjacent_words)
                aw_max = min(len_text_line, i + window_adjacent_words + 1)
                adjacent_words = text_line[aw_min:i] + text_line[i + 1:aw_max]

                nsw_min = max(0, min(aw_min, i - window_close_words))
                nsw_max = min(len_text_line, max(aw_max, i + window_close_words + 1))
                close_words = text_line[nsw_min:aw_min] + text_line[aw_max:nsw_max]

                prob = probabilities_tl[nsw_min:aw_min] + probabilities_tl[aw_max:nsw_max]
                close_words_selected = select_random_labels(close_words, close_words_size, prob)

                context = adjacent_words + close_words_selected
                for label in context:
                    yield label, word


def select_random_labels(labels, num_labels, probabilities):
    if len(labels) <= num_labels:
        return labels
    samples = []
    probabilities_copy = list(probabilities)
    probabilities_sum = np.sum(probabilities_copy)
    for _ in range(num_labels):
        r = random.random() * probabilities_sum
        p_sum = 0.0
        i = 0
        while p_sum < r:
            p_sum += probabilities_copy[i]
            i += 1
        i -= 1
        samples.append(labels[i])
        probabilities_sum -= probabilities_copy[i]
        probabilities_copy[i] = 0.0
    return samples


def data_generator_buffered(data_generator, buffer_size=100000, randomize=True):
    buffer = []
    try:
        while len(buffer) < buffer_size:
            buffer.append(data_generator.next())
    except StopIteration:
        pass
    while len(buffer) > 1:
        if randomize:
            random_pos = random.randrange(len(buffer))
        else:
            random_pos = 0
        yield buffer[random_pos]
        del buffer[random_pos]
        try:
            buffer.append(data_generator.next())
        except StopIteration:
            pass
    yield buffer[0]


def generate_batch(data_generator, batch_size):
    batch = []
    for sample in data_generator:
        batch.append(sample)
        if len(batch) == batch_size:
            inputs, outputs = zip(*batch)
            yield np.asarray(inputs), np.asarray(outputs)
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


def model(batch_size, vocabulary_size, embedding_size, num_negative_samples,
          learning_rate_initial, learning_rate_decay, learning_rate_decay_steps):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.placeholder(tf.int32, name='global_step')
        input_label = tf.placeholder(tf.int32, shape=[batch_size])
        output_word = tf.placeholder(tf.int32, shape=[batch_size])

        input_label_reshaped = tf.reshape(input_label, [batch_size])
        output_word_reshaped = tf.reshape(output_word, [batch_size, 1])

        matrix_dimension = [vocabulary_size, embedding_size]
        embeddings = tf.Variable(tf.random_uniform(matrix_dimension, -1.0, 1.0), name='embeddings')
        embed = tf.nn.embedding_lookup(embeddings, input_label_reshaped)

        # Construct the variables for the NCE loss
        stddev = 1.0 / math.sqrt(embedding_size)
        nce_weights = tf.Variable(tf.truncated_normal(matrix_dimension, stddev=stddev))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                  labels=output_word_reshaped,
                                  inputs=embed, num_sampled=num_negative_samples,
                                  num_classes=vocabulary_size)
        loss = tf.reduce_mean(nce_loss)
        # learning rate & optimizer
        learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step,
                                                   learning_rate_decay_steps, learning_rate_decay,
                                                   staircase=True, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # summaries and moving average
        ema = tf.train.ExponentialMovingAverage(0.9)
        ema_assign = ema.apply([loss])
        with tf.control_dependencies([optimizer]):
            training_op = tf.group(ema_assign)
        average_loss = ema.average(loss)
        tf.summary.scalar('loss', loss)
        # initializing function
        init = tf.global_variables_initializer()

    return graph, global_step, input_label, output_word, init, loss, average_loss, training_op, learning_rate, embeddings


if __name__ == '__main__':
    epochs = 1  # iterations over the whole dataset
    batch_size = 128  # batch size for the training
    embedding_size = 128  # embedding size
    window_adjacent_words = 1  # adjacent words to be added to the context
    close_words_size = 2  # close words (non-adjacent) to be added to the context
    window_close_words = 6  # maximum distance between the target word and the close words
    negative_num_samples = 64  # Number of negative examples to sample.
    learning_rate_initial = 0.1
    learning_rate_decay = 0.928
    learning_rate_decay_steps = 10000

    print('Loading dataset...')
    symbols_dict, encoded_text, word_frequency_dict = load_word2vec_data(
        'data/generated/word2vec_dataset')
    vocabulary_size = len(set(symbols_dict.values()))

    print('Buffering data...')
    data_generator = generate_training_samples(encoded_text, epochs, window_adjacent_words,
                                               close_words_size, window_close_words,
                                               word_frequency_dict)
    data_generator = data_generator_buffered(data_generator)

    print('Creating graph...')
    graph, global_step, input_label, output_word, init, loss, average_loss, optimizer, learning_rate, embeddings = \
        model(batch_size, vocabulary_size, embedding_size, negative_num_samples,
              learning_rate_initial, learning_rate_decay, learning_rate_decay_steps)

    print('Training...')
    t1 = time.time()
    # set tensorflow to not to use all memory so you can run multiple process on the gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(graph=graph, config=config) as session:
        init.run()
        step = 0
        try:
            for words, labels in generate_batch(data_generator, batch_size):
                lr, _, _, loss_val = session.run([learning_rate, optimizer, loss, average_loss],
                                                 feed_dict={
                                                     global_step: step,
                                                     input_label: labels,
                                                     output_word: words,
                                                 })
                step += 1
                if step % 100000 == 0:
                    elapsed_time = str(timedelta(seconds=time.time() - t1))
                    message = 'step: {}  loss: {:0.4f}  learning_rate = {:0.6f}  elapsed: {}'
                    print(message.format(step, loss_val, lr, elapsed_time))
        except StopIteration:
            pass

        print('Saving embeddings...')
        embeddings_eval = embeddings.eval()
    norm = np.sqrt(np.sum(np.square(embeddings_eval)))
    normalized_embeddings = embeddings_eval / norm
    with open('data/generated/embeddings_{}'.format(embedding_size), 'wb') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(normalized_embeddings)
