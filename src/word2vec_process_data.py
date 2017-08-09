from configuration import *
import re
import os
from preprocess_data import load_csv_wikipedia_gen, load_csv_dataset, group_count


def load_word2vec_data(filename, vocabulary_size=VOCABULARY_SIZE):
    """
    Loads the word2vec data: the dictionary file with the relation of the word with its int id,
    the dataset as a list of list of ids and a dictionary with the frequency of the words in
    the dataset.
    :param str filename: name of the file with the word2vec dataset, the dictionary file and the
    frequency file are generated with the suffixes _dict and _count based on this fiename
    :return (Dict[str,int], List[List[int]], Dict[int,float]: a tuple with a dictionary for the
    symbols and a list of sentences where each sentence is a list of int and a dictionary with
    the frequencies of the words
    """
    filename = '{}_{}'.format(filename, vocabulary_size)
    filename_dict = '{}_dict'.format(filename)
    filename_count = '{}_count'.format(filename)
    with open(os.path.join(DIR_DATA_WORD2VEC, filename_dict), 'r') as f:
        symbols_dict = {}
        for line in f.readlines():
            data = line.split()
            symbol = data[0]
            encoded = int(data[1])
            symbols_dict[symbol] = encoded
    encoded_text = []
    with open(os.path.join(DIR_DATA_WORD2VEC, filename), 'r') as f:
        for line in f.readlines():
            encoded_text.append([int(word) for word in line.split()])
    total_count = 0
    with open(os.path.join(DIR_DATA_WORD2VEC, filename_count), 'r') as f:
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


def load_or_create_dataset_word2vec(filename, text_samples, vocabulary_size=VOCABULARY_SIZE):
    """
    Loads the dataset for word2vec or creates it from the text_samples if the file doesn't exits.
    Three files are generated: dictionary file, word frequency file and dataset file. The dataset
    file already contains the ids instead of the words. The vocabulary is truncated to fit the
    vocabulary size, the less frequent words are transformed into the unknown id (the number 0)
    :param str filename: filename prefix of the dataset
    :param List[List[str]] text_samples: list of list of words
    :param int vocabulary_size: the final size of the vocabulary
    :return (Dict[str,int], List[List[int]], Dict[int,float]: a tuple with a dictionary for the
    symbols and a list of sentences where each sentence is a list of int and a dictionary with
    the frequencies of the words
    """
    filename_vocabulary = '{}_{}'.format(filename, vocabulary_size)
    filename_dict = '{}_dict'.format(filename_vocabulary)
    filename_count = '{}_count'.format(filename_vocabulary)
    if not os.path.exists(os.path.join(DIR_DATA_WORD2VEC, filename_vocabulary)):
        text_lines = []
        for text_sample in text_samples:
            sentences = re.split('\n|\s\.', text_sample.lower())
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 0:
                    words.append('.')
                    words = list([word.strip() for word in words])
                    text_lines.append(words)
        symbols_count = group_count(text_lines)
        symbols_ordered_by_count = sorted(symbols_count.items(), key=lambda x: x[1], reverse=True)
        total_symbols = len(symbols_ordered_by_count)
        print('Total symbols: {}'.format(total_symbols))
        print('Vocabulary size: {}'.format(vocabulary_size))
        unknown_symbols = symbols_ordered_by_count[vocabulary_size - 1:]
        known_symbols = symbols_ordered_by_count[:vocabulary_size - 1]
        symbols_dict = {}
        for symbol, _ in unknown_symbols:
            symbols_dict[symbol] = 0
        counter = 1
        for symbol, _ in known_symbols:
            symbols_dict[symbol] = counter
            counter += 1
        encoded_text = []

        words_count = 0
        for sentence in text_lines:
            words_count += len(sentence)
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.append(symbols_dict[word])
            if len(encoded_sentence) > 0:
                encoded_text.append(encoded_sentence)
        print('Total sentences: {}'.format(len(text_lines)))
        print('Total words: {}'.format(words_count))
        print('words/sentences: {}'.format(float(words_count) / float(len(text_lines))))

        with open(os.path.join(DIR_DATA_WORD2VEC, filename_dict), 'wb') as f:
            for symbol in sorted(symbols_dict.keys()):
                f.write('{} {}\n'.format(symbol, symbols_dict[symbol]))
        with open(os.path.join(DIR_DATA_WORD2VEC, filename_vocabulary), 'wb') as f:
            for sentence in encoded_text:
                f.write(' '.join(str(word) for word in sentence))
                f.write('\n')
        with open(os.path.join(DIR_DATA_WORD2VEC, filename_count), 'wb') as f:
            for symbol, count in symbols_ordered_by_count:
                f.write('{} = {}\n'.format(symbol, count))

    return load_word2vec_data(filename)


if __name__ == '__main__':
    print('Generate text for Word2Vec model... (without using test data)')
    train_set = load_csv_dataset('train_set_numbers_parsed')
    genes_articles = load_csv_wikipedia_gen('wikipedia_mutations_parsed')
    word2vec_text = [s.text for s in genes_articles] + [s.text for s in train_set]
    symbols_dict, word2vec_encoded_text, word_frequency = \
        load_or_create_dataset_word2vec('word2vec_dataset', word2vec_text)
