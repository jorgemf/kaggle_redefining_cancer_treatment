# coding=utf-8
from .preprocess_data import *
from .text_classification_process_data import *

def _filter_clear_val_set(val_set):
    with open(os.path.join(DIR_DATA, 'stage1_solution_filtered.csv'), 'rb') as file:
        reader = csv.reader(file)
        next(reader, None) # skip header
        stage1_test_classes = {}
        for row in reader:
            row = [int(r) for r in row]
            stage1_test_classes[row[0]] = row[1:].index(1) + 1
    val_set_filter = []
    for d in val_set:
        if d.id in stage1_test_classes:
            d.real_class = stage1_test_classes[d.id]
            val_set_filter.append(d)
    return val_set_filter

if __name__ == '__main__':
    print('Load raw data...')
    train_set = load_raw_dataset('training_text', 'training_variants', ignore_empty=True)
    val_set = load_raw_dataset('test_text', 'test_variants')
    val_set = _filter_clear_val_set(val_set)
    stage2_test_set = load_raw_dataset('stage2_test_text.csv', 'stage2_test_variants.csv')
    print('Clean raw data or load already clean data...')
    train_set = load_or_clean_text_dataset('train_set_text_clean', train_set)
    val_set = load_or_clean_text_dataset('val_set_text_clean', val_set)
    stage2_test_set = load_or_clean_text_dataset('stage2_test_set_text_clean', stage2_test_set)
    genes = set([s.gene for s in train_set] + [s.gene for s in val_set])
    variations = set([s.variation for s in train_set] + [s.variation for s in val_set])
    if not all(is_mutation(word, genes) for word in variations):
        wrong_detections = sorted(
                set([word.strip() for word in variations if not is_mutation(word, genes)]))
        print('WARNING not all variations are detected as mutations: {}'.format(
                ", ".join(wrong_detections)))
    print('Tokenizer with nltk...')
    nltk.download('punkt')
    tokenize_documents(stage2_test_set)
    tokenize_documents(val_set)
    print('Parse mutations to tokens...')
    stage2_test_set = load_or_parse_mutations_dataset('stage2_test_set_mutations_parsed',
                                                      stage2_test_set, genes)
    val_set = load_or_parse_mutations_dataset('val_set_mutations_parsed',
                                              val_set, genes)
    print('Parse numbers to tokens...')
    stage2_test_set = load_or_parse_numbers_dataset('stage2_test_set_numbers_parsed',
                                                    stage2_test_set)
    val_set = load_or_parse_numbers_dataset('val_test_set_numbers_parsed', val_set)
    print('Transform words into ids')
    word_dict = load_word2vec_dict('word2vec_dataset')
    transform_words_in_ids(stage2_test_set, word_dict)
    transform_words_in_ids(val_set, word_dict)
    print('Generating samples for stage2 test set...')
    save_text_classification_dataset('stage2_test_set', stage2_test_set)
    save_text_classification_dataset('stage2_test_set', stage2_test_set, dir=DIR_DATA_DOC2VEC)
    save_text_classification_dataset('val_set', val_set)
    save_text_classification_dataset('val_set', val_set, dir=DIR_DATA_DOC2VEC)
