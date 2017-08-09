import os

# directories for the data

DIR_DATA = 'data'
DIR_GENERATED_DATA = os.path.join(DIR_DATA, 'generated')
DIR_DATA_WORD2VEC = os.path.join(DIR_GENERATED_DATA, 'word2vec')
DIR_DATA_TEXT_CLASSIFICATION = os.path.join(DIR_GENERATED_DATA, 'text_classification')
DIR_W2V_LOGDIR = os.path.join('.', 'tmp', 'logdir', 'word2vec')
DIR_TC_LOGDIR = os.path.join('.', 'tmp', 'logdir', 'text_classification')
LOCAL_REPO = ''  # TODO
LOCAL_REPO_W2V_PATH = 'word2vec'
LOCAL_REPO_TC_PATH = 'text_classification'

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
W2V_LEARNING_RATE_INITIAL = 0.1  # Initial learning rate for gradient descent
W2V_LEARNING_RATE_DECAY = 0.928  # Decay of learning rate
W2V_LEARNING_RATE_DECAY_STEPS = 10000  # Steps to decay the learnin rate
W2V_DATA_BUFFER_SIZE = 100000  # size of the buffer to randomize the input data for training

# text classification

TC_EPOCHS = 2  # iterations over the whole dataset
TC_BATCH_SIZE = 32  # batch size for the training
