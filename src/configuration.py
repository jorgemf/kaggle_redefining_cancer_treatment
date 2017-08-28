import os
from trainer import get_data_path

# directories for the data

DIR_DATA = 'data'
DIR_GENERATED_DATA = get_data_path(dataset_name='jorgemf/kaggle-redefining-cancere-treatment',
                                   local_root=DIR_DATA,
                                   local_repo='generated')
DIR_DATA_WORD2VEC = os.path.join(DIR_GENERATED_DATA, 'word2vec')
DIR_DATA_DOC2VEC = os.path.join(DIR_GENERATED_DATA, 'doc2vec')
DIR_DATA_TEXT_CLASSIFICATION = os.path.join(DIR_GENERATED_DATA, 'text_classification')
# log dirs routes are automatically handle in the trainer
DIR_W2V_LOGDIR = os.path.join('.', 'model', 'train', 'word2vec')
DIR_D2V_LOGDIR = os.path.join('.', 'model', 'train', 'doc2vec')
DIR_TC_LOGDIR = os.path.join('.', 'model', 'train', 'text_classification')

# pre process data

DIR_WIKIPEDIA_GENES = os.path.join(DIR_GENERATED_DATA, 'gen')

# shared conf between word2vec and text_classification models

VOCABULARY_SIZE = 30000
EMBEDDINGS_SIZE = 128
MAX_WORDS = 40000  # maximum number of words in the document
MAX_SENTENCES = 1000  # maximum number of sentences in the document
MAX_WORDS_IN_SENTENCE = 60  # maximum number of words per sentence in the document

# word2vec

W2V_EPOCHS = 6  # iterations over the whole dataset
W2V_BATCH_SIZE = 128  # batch size for the training
W2V_WINDOW_ADJACENT_WORDS = 1  # adjacent words to be added to the context
W2V_CLOSE_WORDS_SIZE = 2  # close words (non-adjacent) to be added to the context
W2V_WINDOW_CLOSE_WORDS = 6  # maximum distance between the target word and the close words
W2V_NEGATIVE_NUM_SAMPLES = 64  # number of negative examples to sample for training
W2V_LEARNING_RATE_INITIAL = 0.01  # initial learning rate for gradient descent
W2V_LEARNING_RATE_DECAY = 0.9  # decay of learning rate
W2V_LEARNING_RATE_DECAY_STEPS = 100000  # steps to decay the learning rate

# doc2vec

D2V_EPOCHS = 3  # iterations over the whole dataset
D2V_BATCH_SIZE = 128  # batch size for the training
D2V_CONTEXT_SIZE = 5  # size of the context to predict the word
D2_TRAIN_DOCS_SIZE = 3321  # number of docs in the train set
D2_TEST_DOCS_SIZE = 5668  # number of docs in the test set
D2V_NEGATIVE_NUM_SAMPLES = 64  # number of negative examples to sample for training
D2V_LEARNING_RATE_INITIAL = 0.01  # initial learning rate for gradient descent
D2V_LEARNING_RATE_DECAY = 0.9  # decay of learning rate
D2V_LEARNING_RATE_DECAY_STEPS = 100000  # steps to decay the learning rate

# text classification

TC_DATA_AUGMENTATION_SAMPLES_PER_CLASS = 2000  # number of samples per class for data augmentation
TD_DATA_SENTENCE_REMOVE_PERCENTAGE = 0.05  # ratio of sentences to delete from the samples
TC_EPOCHS = 10  # iterations over the whole dataset
TC_BATCH_SIZE = 4  # batch size for the training
TC_MODEL_HIDDEN = 200  # hidden GRUCells for the model
TC_MODEL_LAYERS = 3  # number of layers of the model
TC_MODEL_DROPOUT = 0.8  # dropout during training in the model
TC_LEARNING_RATE_INITIAL = 0.001  # initial learning rate for gradient descent
TC_LEARNING_RATE_DECAY = 0.9  # decay of learning rate
TC_LEARNING_RATE_DECAY_STEPS = 1000  # steps to decay the learning rate
TC_CNN_FILTERS = 128  # number of dimensions of the cnn network
TC_CNN_LAYERS = 2  # number of layers of the cnn network
TC_HATT_WORD_OUTPUT_SIZE = 128  # number of words outputs size for the hatt model
TC_HATT_SENTENCE_OUTPUT_SIZE = 128  # number of sentences outputs size for the hatt model
