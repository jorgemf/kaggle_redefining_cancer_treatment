import io
from src.text_classification_process_data import save_text_classification_dataset
from src.text_classification_process_data import transform_words_in_ids
from src.text_classification_process_data import load_word2vec_dict
from src.text_classification_process_data import load_csv_dataset
from src.configuration import *

if __name__ == '__main__':
    print('Generate text for Doc2Vec model...')
    train_set = load_csv_dataset('train_set_numbers_parsed')
    test_set = load_csv_dataset('test_set_numbers_parsed')
    print('{} docs in train set'.format(len(train_set)))
    print('{} docs in test set'.format(len(test_set)))
    print('Transform words into ids')
    word_dict = load_word2vec_dict('word2vec_dataset')
    transform_words_in_ids(train_set, word_dict)
    transform_words_in_ids(test_set, word_dict)
    print('Saving final dataset...')
    save_text_classification_dataset('train_set', train_set, dir=DIR_DATA_DOC2VEC)
    save_text_classification_dataset('test_set', test_set, dir=DIR_DATA_DOC2VEC)
    print('Creating tsv file for tensorboard...')
    with io.open(os.path.join(DIR_DATA_DOC2VEC, 'train_set_classes.tsv'), 'w',
                 encoding='utf8') as f:
        f.write('class\tid\n')
        pos = 0
        for sample in train_set:
            f.write(u'{}\t{}\n'.format(sample.real_class, pos))
            pos += 1
