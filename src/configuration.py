import os
from trainer import get_data_path

# directories for the data

DIR_DATA = 'data'
DIR_GENERATED_DATA = get_data_path(dataset_name='jorgemf/kaggle-redefining-cancere-treatment',
                                   local_root=DIR_DATA,
                                   local_repo='generated')
DIR_DATA_WORD2VEC = os.path.join(DIR_GENERATED_DATA, 'word2vec')
DIR_DATA_TEXT_CLASSIFICATION = os.path.join(DIR_GENERATED_DATA, 'text_classification')
# log dirs routes are automatically handle in the trainer
DIR_W2V_LOGDIR = os.path.join('.', 'model', 'train', 'word2vec')
DIR_TC_LOGDIR = os.path.join('.', 'model', 'train', 'text_classification')

# pre process data

DIR_WIKIPEDIA_GENES = os.path.join(DIR_GENERATED_DATA, 'gen')

# shared conf between word2vec and text_classification models

VOCABULARY_SIZE = 30000
EMBEDDINGS_SIZE = 128

# word2vec

W2V_EPOCHS = 1  # iterations over the whole dataset
W2V_BATCH_SIZE = 128  # batch size for the training
W2V_WINDOW_ADJACENT_WORDS = 1  # adjacent words to be added to the context
W2V_CLOSE_WORDS_SIZE = 2  # close words (non-adjacent) to be added to the context
W2V_WINDOW_CLOSE_WORDS = 6  # maximum distance between the target word and the close words
W2V_NEGATIVE_NUM_SAMPLES = 64  # number of negative examples to sample.
W2V_LEARNING_RATE_INITIAL = 0.1  # initial learning rate for gradient descent
W2V_LEARNING_RATE_DECAY = 0.928  # decay of learning rate
W2V_LEARNING_RATE_DECAY_STEPS = 10000  # steps to decay the learning rate
W2V_DATA_BUFFER_SIZE = 100000  # size of the buffer to randomize the input data for training

# text classification

TC_DATA_AUGMENTATION_SAMPLES_PER_CLASS = 2000  # number of samples per class for data augmentation
TD_DATA_SENTENCE_REMOVE_PERCENTAGE = 0.05  # ratio of sentences to delete from the samples
TC_EPOCHS = 10  # iterations over the whole dataset
TC_BATCH_SIZE = 32  # batch size for the training
TC_MODEL_HIDDEN = 200  # hidden GRUCells for the model
TC_MODEL_LAYERS = 3  # number of layers of the model
TC_MODEL_DROPOUT = 0.8  # dropout during training in the model
TC_LEARNING_RATE_INITIAL = 0.1  # initial learning rate for gradient descent
TC_LEARNING_RATE_DECAY = 0.9  # decay of learning rate
TC_LEARNING_RATE_DECAY_STEPS = 1000  # steps to decay the learning rate
