import numpy as np
import io
import random
from .preprocess_data import load_csv_dataset
from .configuration import *


def load_word2vec_dict(filename, vocabulary_size=VOCABULARY_SIZE):
    """
    Loads the word2vec dictionary
    :param str filename: name of the file with the dict
    :param int vocabulary_size: size of the vocabulary
    :return Dict[str, int]: Dictionary where the key is the word and teh value is the encoded value
    for the word as an integer.
    """
    filename = '{}_{}'.format(filename, vocabulary_size)
    filename_dict = '{}_dict'.format(filename)
    with io.open(os.path.join(DIR_DATA_WORD2VEC, filename_dict), 'r', encoding='utf8') as f:
        symbols_dict = {}
        for line in f.readlines():
            data = line.split()
            symbol = data[0]
            encoded = int(data[1])
            symbols_dict[symbol] = encoded
    return symbols_dict


def transform_words_in_ids(dataset, symbols_dict):
    """
    Uses a dictionary of symbols to translate the string words into encoded integers for the text
    in the dataset
    :param List[DataSample] dataset: dataset of DataSample
    :param Dict[str, int] symbols_dict: dictionary with the encoded values of the words
    """
    for datasample in dataset:
        sentences = datasample.text.split(' . ')
        parsed_sentences = []
        for sentence in sentences:
            encoded_sentence = []
            words = sentence.split()
            if len(words) > 0:
                words.append('.')
                words = list([word.strip().lower() for word in words])
                for word in words:
                    if word not in symbols_dict:
                        print(u'word "{}" not in dict, parsed to unknown token 0'.format(word))
                        encoded_sentence.append(0)
                    else:
                        encoded_sentence.append(symbols_dict[word.lower()])
            if len(encoded_sentence) > 0:
                parsed_sentences.append(encoded_sentence)
        datasample.text = parsed_sentences


def balance_class(dataset):
    """
    Balance the classes to a target value of number of elements per class. This method only
    replicates the samples in the class until if matchs the final_num
    :param List[DataSample] dataset: dataset of DataSample
    :param final_num: target number of elements per class
    :return List[DataSample]: the new dataset with the repeated samples. It is also shuffled
    """
    classes_group = {}
    for d in dataset:
        if d.real_class not in classes_group:
            classes_group[d.real_class] = []
        classes_group[d.real_class].append(d)
    classes_string = ", ".join(
        ["{}:{}".format(k, len(classes_group[k])) for k in sorted(classes_group.keys())])
    print("{} different classes: {}".format(len(classes_group), classes_string))
    max_in_class = np.max([len(v) for v in classes_group.values()]) * 2
    new_dataset = []
    for key, class_list in classes_group.iteritems():
        random.shuffle(class_list)
        for index in range(max_in_class - len(class_list)):
            class_list.append(class_list[index].__copy__())
        new_dataset.extend(class_list)

    random.shuffle(new_dataset)
    return new_dataset


def remove_random_sentences(dataset, ratio_to_remove=TD_DATA_SENTENCE_REMOVE_PERCENTAGE):
    """
    Removes random sentences of the text in the dataset. This method along with balance_class are a
    way to augmentate the dataset.
    :param List[DataSample] dataset: dataset of DataSample
    :param float ratio_to_remove: ratio of sentences to be removed from each sample in the dataset
    :return List[DataSample]: the dataset with the removed sentences
    """
    for sample in dataset:
        to_remove = int(len(sample.text) * ratio_to_remove)
        text = sample.text
        for _ in range(to_remove):
            text.pop(random.randint(0, len(text) - 1))
    return dataset


def save_text_classification_dataset(filename, dataset, dir=DIR_DATA_TEXT_CLASSIFICATION):
    """
    Saves the dataset. The sentences are stored in one single line, so they can processed better
    when they are read for training or test
    :param str filename: filename where to store the dataset
    :param List[DataSample] dataset: the dataset of DataSample
    """
    with open(os.path.join(dir, filename), 'wb') as file:
        for data in dataset:
            file.write('{} '.format(data.real_class))
            for sentence in data.text:
                for word in sentence:
                    file.write('{} '.format(word))
            file.write('\n')


def data_stats(train_set, test_set):
    """
    Show statistics of the datasets
    :param train_set:
    :param test_set:
    """
    sum_docs_lengths = 0
    sum_sentences_lengths = 0
    num_sentences = 0
    max_doc_length = 0
    max_sentence_length = 0
    max_sentences_doc = 0
    num_docs = 0
    for data in train_set + test_set:
        num_docs += 1
        dwl = 0
        dsl = len(data.text)
        num_sentences += dsl
        if dsl > max_sentences_doc:
            max_sentences_doc = dsl
        for sentence in data.text:
            sl = len(sentence)
            dwl += sl
            sum_sentences_lengths += sl - 1  # remove the last dot
            if sl > max_sentence_length:
                max_sentence_length = sl
        sum_docs_lengths += dwl
        if dwl > max_doc_length:
            max_doc_length = dwl

    print('Average words in a doc is {}'.format(sum_docs_lengths / num_docs))
    print('Maximum words in a doc is {}'.format(max_doc_length))
    print('Average sentences in a doc is {}'.format(num_sentences / num_docs))
    print('Maximum sentences in a doc is {}'.format(max_sentences_doc))
    print('Average sentence is {}'.format(sum_sentences_lengths / num_sentences))
    print('Longest sentence is {}'.format(max_sentence_length))
    # Average words in a doc is 12617
    # Maximum words in a doc is 115316
    # Average sentences in a doc is 340
    # Maximum sentences in a doc is 3282
    # Average sentence is 36
    # Longest sentence is 4048

if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.INFO)
    print('Generate text data augmentation for text classification model...')
    train_set = load_csv_dataset('train_set_numbers_parsed')
    test_set = load_csv_dataset('test_set_numbers_parsed')
    print('Transform words into ids')
    word_dict = load_word2vec_dict('word2vec_dataset')
    transform_words_in_ids(train_set, word_dict)
    transform_words_in_ids(test_set, word_dict)
    print('Calculating statistics...')
    data_stats(train_set, test_set)
    print('Balancing classes...')
    train_set = balance_class(train_set)
    print('Removing random sentences...')
    train_set = remove_random_sentences(train_set)
    print('Saving final training dataset...')
    save_text_classification_dataset('train_set', train_set)
    print('Generating samples for test set...')
    save_text_classification_dataset('test_set', test_set)
