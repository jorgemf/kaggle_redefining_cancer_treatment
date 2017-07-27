import zipfile
import re
import os
import csv
import sys
import urllib2
from bs4 import BeautifulSoup
import unicodedata

csv.field_size_limit(sys.maxsize)


def extract_zip_file(filepath, directory):
    zip_ref = zipfile.ZipFile(filepath, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()


def extract_zip_files():
    files = ['data/training_text', 'data/training_variants', 'data/test_text', 'data/test_variants']
    for file in files:
        if not os.path.exists(file):
            extract_zip_file('{}.zip'.format(file), 'data/')


def load_raw_dataset(text_file, variants_file):
    with open(text_file) as file:
        lines = file.readlines()
        data_text = lines[1:]  # ignore header
    with open(variants_file) as file:
        lines = file.readlines()
        data_variant = lines[1:]  # ignore header
        header_variants = lines[0].split(',')
    data = []
    for dataline_text, dataline_variant in zip(data_text, data_variant):
        dataline_text_split = dataline_text.split('||')
        dataline_variant_split = dataline_variant.split(',')
        # basic checks
        if len(dataline_text_split) != 2:
            raise Exception('error in text file in line {}'.format(dataline_text))
        if len(dataline_variant_split) < 3 or len(dataline_variant_split) > 4:
            raise Exception('error in variant file in line {}'.format(dataline_variant))
        if dataline_text_split[0] != dataline_variant_split[0]:
            raise Exception(
                'wrong ids in text and variant files {} !+ {}'.format(dataline_text_split[0],
                                                                      dataline_variant_split[0]))
        id = int(dataline_text_split[0])
        text = dataline_text_split[1]
        gene = dataline_variant_split[1]
        variation = dataline_variant_split[2]
        if len(header_variants) == 4:
            real_class = int(dataline_variant_split[3])
        else:
            real_class = None
        data.append([id, text, gene, variation, real_class])
    return data


def load_raw_data():
    train_set = load_raw_dataset('data/training_text', 'data/training_variants')
    test_set = load_raw_dataset('data/test_text', 'data/test_variants')
    return train_set, test_set


def load_csv(filename):
    with open(filename) as file:
        reader = csv.reader(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        return list([row for row in reader])


def save_csv(filename, dataset):
    with open(filename, 'wb') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(dataset)


def get_genes(dataset):
    return [s[2] for s in dataset]


def get_variations(dataset):
    return [s[3] for s in dataset]


def get_text(dataset):
    return [s[1] for s in dataset]


RE_BIBLIOGRAPHIC_REFERENCE_1 = re.compile(r"\s*\[[\d\s,]+\]\s*")
RE_BIBLIOGRAPHIC_REFERENCE_2 = re.compile(r"\s*\(([a-zA-Z\s\.,]+\d{2,4};?)+\s*\)\s*")
RE_BIBLIOGRAPHIC_REFERENCE_3 = re.compile(r"\s*\([\d,\s]+\)\s*")
RE_BIBLIOGRAPHIC_REFERENCE_4 = re.compile(r"\s*(\w+ et al\.,?)+")

RE_FIGURES = re.compile(r"\s*(Fig(ure)?\.? [\w,]+)\s*")
RE_TABLES = re.compile(r"\s*(Table\.? [\w,]+)\s*")
RE_WHITE_SPACES = re.compile(r"\s+")
RE_EMTPY_PARENTHESES = re.compile(r"\(\s*(and)?\s*\)")


def clean_text(text):
    # remove bibliographic references
    text = re.sub(RE_BIBLIOGRAPHIC_REFERENCE_1, ' ', text)
    text = re.sub(RE_BIBLIOGRAPHIC_REFERENCE_2, ' ', text)
    text = re.sub(RE_BIBLIOGRAPHIC_REFERENCE_3, ' ', text)
    text = re.sub(RE_BIBLIOGRAPHIC_REFERENCE_4, ' ', text)
    # remove figures
    text = re.sub(RE_FIGURES, "", text)
    # remove tables
    text = re.sub(RE_TABLES, "", text)
    # remove empty parentheses
    text = re.sub(RE_EMTPY_PARENTHESES, "", text)
    # add white spaces before and after symbols
    text = text.replace('...', '.')
    for symbol in ['(', ')', '/', '-', '\xe2', '\'', '%', ':', '?', ', ', '. ', '<', '>', '=', '-']:
        text = text.replace(symbol, " {} ".format(symbol))
    # remove double white spaces
    text = re.sub(RE_WHITE_SPACES, ' ', text)
    return text


def group_count(elements, group={}):
    for e in elements:
        if isinstance(e, list):
            group = group_count(e, group)
        elif e in group:
            group[e] += 1
        else:
            group[e] = 1
    return group


def show_stats(train_set, test_set):
    print("{} samples in the training set".format(len(train_set)))
    print("{} samples in the test set".format(len(test_set)))
    classes = [d[4] for d in train_set]
    classes_group = group_count(classes)
    classes_string = ", ".join(
        ["{}:{}".format(k, classes_group[k]) for k in sorted(classes_group.keys())])
    print("{} different classes: {}".format(len(set(classes)), classes_string))
    train_genes = get_genes(train_set)
    test_genes = get_genes(test_set)
    print("{} genes in training set".format(len(set(train_genes))))
    print("{} genes in test set".format(len(set(test_genes))))
    print("{} genes in test and train set".format(len(set(test_genes + train_genes))))


def get_genes_articles_from_wikipedia(genes):
    data = []
    for gen in genes:
        filename = 'data/generated/wikipedia_gen_{}'.format(gen)
        if not os.path.exists(filename):
            url = 'https://en.wikipedia.org/wiki/{}'.format(gen)
            try:
                html = BeautifulSoup(urllib2.urlopen(url).read(), 'lxml')
                html_data = html.find(id='mw-content-text').div.find_all('p')
                text_data = [h.get_text() for h in html_data]
                text_data = [t.strip() for t in text_data if
                             len(t.strip()) > 30 and len(t.strip().split()) > 10]
                text_data = [unicodedata.normalize('NFKD', l).encode('ascii', 'ignore') for l in
                             text_data]
            except:
                text_data = ['']
            with open(filename, 'wb') as f:
                f.writelines(text_data)
        with open(filename, 'r') as f:
            text_lines = f.readlines()
            text = "\n".join(text_lines)
        data.append([gen, text])
    return data


def load_or_clean_text_dataset(filename, dataset):
    if not os.path.exists(filename):
        for datasample in dataset:
            datasample[1] = clean_text(datasample[1])
        save_csv(filename, dataset)
    return load_csv(filename)


def load_or_parse_mutations_dataset(filename, dataset, genes):
    if not os.path.exists(filename):
        for datasample in dataset:
            words = datasample[1].split()
            parsed_words = []
            for word in words:
                if is_mutation(word, genes):
                    parsed_words.extend(split_mutation(word))
                else:
                    parsed_words.append(word)
            datasample[1] = ' '.join(parsed_words)
        save_csv(filename, dataset)
    return load_csv(filename)


def encode_number(number):
    if number < 0.001:
        return '>number_001'
    elif number < 0.1:
        return '>number_01'
    elif number < 1.0:
        return '>number_1'
    elif number < 10.0:
        return '>number_10'
    elif number < 100.0:
        return '>number_100'
    else:
        return '>number_1000'


def load_or_parse_numbers_dataset(filename, dataset):
    if not os.path.exists(filename):
        for datasample in dataset:
            words = datasample[1].split()
            parsed_words = []
            for word in words:
                try:
                    number = float(word)
                    parsed_words.append(encode_number(number))
                except ValueError:
                    parsed_words.append(word)
            datasample[1] = ' '.join(parsed_words)
        save_csv(filename, dataset)
    return load_csv(filename)


def is_mutation(word, genes):
    word = word.strip()
    if len(word) >= 3 and word not in genes:
        has_hyphen_minus = '_' in word
        has_hyphen = '-' in word
        has_digits = any(ch.isdigit() for ch in word)
        has_three_digits = sum(1 for ch in word if ch.isdigit()) > 2
        has_upper_case = any(ch.isupper() for ch in word)
        has_two_upper_case = sum(1 for ch in word if ch.isupper()) > 1
        has_lower_case = any(ch.islower() for ch in word)
        has_symbols = any(not ch.isalnum() for ch in word)
        return has_hyphen_minus or \
               (has_digits and has_two_upper_case) or \
               (has_three_digits and has_upper_case) or \
               (has_digits and has_upper_case and has_symbols) or \
               (has_digits and has_lower_case) or \
               (has_hyphen and has_two_upper_case) or \
               (has_lower_case and has_two_upper_case)
    return False


def split_mutation(word):
    word = word.strip()
    for symbol in ['del', 'ins', 'dup', 'trunc', 'splice', 'fs', 'null', 'Fusion']:
        word = word.replace(symbol, ' >{} '.format(symbol))
    i = 0
    new_words = []
    while i < len(word):
        if word[i] == '>':
            j = i + 1
            while j < len(word) and word[j] != ' ':
                j += 1
            new_words.append('{}'.format(word[i:j]))
            i = j
        elif word[i] != ' ':
            new_words.append('>{}'.format(word[i]))
            i += 1
        else:
            i += 1
    return new_words

def load_or_create_dataset_word2vec(filename, text_samples, vocabulary_size=30000):
    if not os.path.exists('{}_{}'.format(filename, vocabulary_size)):
        text_lines = []
        for text_sample in text_samples:
            sentences = re.split('\n|\s\.', text_sample.lower())
            for sentence in sentences:
                words = sentence.split()
                words = list([word.strip() for word in words])
                text_lines.append(words)
        symbols_count = group_count(text_lines)
        symbols_ordered_by_count = sorted(symbols_count.items(), key=lambda x: x[1], reverse=True)
        total_symbols = len(symbols_ordered_by_count)
        print('Total symbols: {}'.format(total_symbols))
        unknown_symbols = symbols_ordered_by_count[vocabulary_size:]
        known_symbols = symbols_ordered_by_count[:vocabulary_size]
        symbols_dict = {}
        for symbol in unknown_symbols:
            symbols_dict[symbol] = 0
        counter = 1
        for symbol in known_symbols:
            symbols_dict[symbol] = counter
            counter += 1
        encoded_text = []

        words_count = 0
        for sentence in text_lines:
            words_count += len(sentence)
            encoded_sentence = []
            for word in sentence:
                encoded_sentence.append(symbols_dict[word])
            encoded_text.append(encoded_sentence)
        print('Total sentences: {}'.format(len(text_lines)))
        print('Total words: {}'.format(words_count))

        with open(filename, 'wb') as f:
            f.writelines(text_lines)
    with open(filename, 'r') as f:
            text_lines = f.readlines()
    return symbols_dict, encoded_text



    for datasample in dataset:
        all_text.extend(datasample[1].split(' '))
    symbols = group_count(all_text)
    # TODO

if __name__ == '__main__':
    print('Extract zip files if not already done...')
    extract_zip_files()
    print('Load raw data...')
    train_set, test_set = load_raw_data()
    print('Clean raw data or load already clean data...')
    train_set = load_or_clean_text_dataset('data/generated/train_set_text_clean', train_set)
    test_set = load_or_clean_text_dataset('data/generated/test_set_text_clean', test_set)
    print('Statistics about the data:')
    show_stats(train_set, test_set)
    genes = set(get_genes(train_set) + get_genes(test_set))
    variations = set(get_variations(train_set) + get_variations(test_set))
    if not all(is_mutation(word, genes) for word in variations):
        wrong_detections = sorted(
            set([word.strip() for word in variations if not is_mutation(word, genes)]))
        print('WARNING not all variations are detected as mutations: {}'.format(
            ", ".join(wrong_detections)))
    print('Parse mutations to tokens...')
    train_set = load_or_parse_mutations_dataset('data/generated/train_set_mutations_parsed',
                                                train_set, genes)
    test_set = load_or_parse_mutations_dataset('data/generated/test_set_mutations_parsed',
                                               test_set, genes)
    print('Parse numbers to tokens...')
    train_set = load_or_parse_numbers_dataset('data/generated/train_set_numbers_parsed', train_set)
    test_set = load_or_parse_numbers_dataset('data/generated/test_set_numbers_parsed', test_set)
    print('Download articles from wikipedia about genes...')
    genes_articles = get_genes_articles_from_wikipedia(genes)
    print('Clean articles from wikipedia or load already clean data...')
    genes_articles = load_or_clean_text_dataset('data/generated/wikipedia_text_clean',
                                                genes_articles)
    print('Parse mutations to tokens from wikipedia articles...')
    genes_articles = load_or_parse_mutations_dataset('data/generated/wikipedia_mutations_parsed',
                                                     genes_articles, genes)
    print('Parse numbers to tokens from wikipedia articles...')
    genes_articles = load_or_parse_numbers_dataset('data/generated/wikipedia_numbers_parsed',
                                                   genes_articles)
    print('Generate text for Word2Vec model... (without using test data)')
    word2vec_text = get_text(genes_articles) + get_text(train_set)
    symbols_dict, word2vec_encoded_text = \
        load_or_create_dataset_word2vec('data/generated/word2vec_dataset', word2vec_text)
