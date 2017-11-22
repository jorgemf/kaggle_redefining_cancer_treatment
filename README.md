> [Read this in medium](https://medium.com/@jorgemf/personalized-medicine-redefining-cancer-treatment-with-deep-learning-f6c64a366fff)

> Disclaimer: This work has been supported by [Good AI](http://goodailab.com) Lab and all the experiments has been trained using their platform [TensorPort](https://tensorport.com).

> [Appendix: How to reproduce the experiments in TensorPort](https://github.com/jorgemf/kaggle_redefining_cancer_treatment#appendix-how-to-reproduce-the-experiments-in-tensorport)

# Personalized Medicine: Redefining Cancer Treatment with deep learning

## Introduction

In this article we want to show you how to apply deep learning to a domain where we are not experts. Usually applying domain information in any problem we can transform the problem in a way that our algorithms work better, but this is not going to be the case. We are going to create a deep learning model for a Kaggle competition: "[Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment)"

The goal of the competition is to classify a document, a paper, into the type of mutation that will contribute to tumor growth. We also have the gene and the variant for the classification. Currently the interpretation of genetic mutations is being done manually, which it is very time consuming task.

We can approach this problem as a text classification problem applied to the domain of medical articles. It is important to highlight the specific domain here, as we probably won't be able to adapt other text classification models to our specific domain due to the vocabulary used.

Another important challenge we are facing with this problem is that the dataset only contains 3322 samples for training. Usually deep learning algorithms have hundreds of thousands of samples for training. We will have to keep our model simple or do some type of data augmentation to increase the training samples.

In the next sections, we will see related work in text classification, including non deep learning algorithms. Next, we will describe the dataset and modifications done before training. We will continue with the description of the experiments and their results. And finally, the conclusions and an appendix of how to reproduce the experiments in TensorPort.

## Related work in text classification

### Non deep learning models

The classic methods for text classification are based on [bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model) and [n-grams](https://en.wikipedia.org/wiki/N-gram). In both cases, sets of words are extracted from the text and are used to train a simple classifier, as it could be [xgboost](https://en.wikipedia.org/wiki/Xgboost) which it is very popular in kaggle competitions.

There are variants of the previous algorithms, for example the [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf–idf), also known as TF–idf, tries to discover which words are more important per each type of document.

### Word2Vector

[Word2Vec](https://arxiv.org/abs/1301.3781) is not an algorithm for text classification but an algorithm to compute vector representations of words from very large datasets. The peculiarity of word2vec is that the words that share common context in the text are vectors located in the same space. For example, countries would be close to each other in the vector space. Another property of this algorithm is that some concepts are encoded as vectors. For example, the gender is encoded as a vector in such way that the next equation is true: "king - male + female = queen", the result of the math operations is a vector very close to "queen".

Using the word representations provided by Word2Vec we can apply math operations to words and so, we can use algorithms like [Support Vector Machines (SVM)](https://arxiv.org/abs/1301.2785v1) or the deep learning algorithms we will see later. 

There are two ways to train a Word2Vec model:
[Continuous Bag-of-Words, also known as CBOW, and the Skip-Gram](https://arxiv.org/abs/1301.3781). Given a context for a word, usually its adjacent words, we can predict the word with the context (CBOW) or predict the context with the word (Skip-Gram). Both algorithms are similar but Skip-Gram seems to [produce better results for large datasets](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

Besides the linear context we described before, another type of context as a [dependency-based context](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf) can be used.

#### Doc2Vec

[Doc2Vector](https://arxiv.org/abs/1405.4053) or Paragraph2Vector is a variation of Word2Vec that can be used for text classification. This algorithm tries to fix the weakness of traditional algorithms that do not consider the order of the words and also their semantics. 

This algorithm is similar to Word2Vec, it also learns the vector representations of the words at the same time it learns the vector representation of the document. It considers the document as part of the context for the words. Once we train the algorithm we can get the vector of new documents doing the same training in these new documents but with the word encodings fixed, so it only learns the vector of the documents. Then we can apply a clustering algorithm or find the closest document in the training set in order to make a prediction.

### Deep learning models

Deep learning models have been applied successfully to different text-related problems like [text translation](https://arxiv.org/abs/1609.08144) or [sentiment analysis](https://arxiv.org/abs/1710.03203). These models seem to be able to extract semantic information that wasn't possible with other techniques.

[Recurrent neural networks (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) are usually used in problems that require to [transform an input sequence into an output sequence](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) or into a probability distribution (like in text classification). RNN usually uses [Long Short Term Memory (LSTM)](http://www.bioinf.jku.at/publications/older/2604.pdf) cells or the recent [Gated Recurrent Units (GRU)](https://arxiv.org/abs/1412.3555). For example, some authors have used LSTM cells in a [generative and discriminative text classifier](https://arxiv.org/abs/1703.01898).

Convolutional Neural Networks (CNN) are deeply used in [image classification](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) due to their properties to extract features, but they also have been applied to [natural language processing (NLP)](https://arxiv.org/abs/1703.03091). Some authors applied them to a [sequence of words](http://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf) and others to a [sequence of characters](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf). CNNs have also been used along with LSTM cells, for example in the [C-LSMT model for text classification](https://arxiv.org/abs/1511.08630).

CNN is not the only idea taken from image classification to sequences. The idea of [residual connections for image classification (ResNet)](https://arxiv.org/abs/1512.03385) has also been applied to sequences in  [Recurrent Residual Learning for Sequence Classification](https://www.aclweb.org/anthology/D16-1093). The depthwise separable convolutions used in [Xception](https://arxiv.org/abs/1610.02357) have also been applied in text translation in [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059).

RNN are usually slow for long sequences with small batch sizes, as the input of a cell depends of the output of other, which limits its parallelism. In order to solve this problem, [Quasi-Recurrent Neural Networks (QRNN)](https://arxiv.org/abs/1611.01576) were created. They alternate convolutional layers with minimalist recurrent pooling.

Recently, some authors have included attention in their models. The attention mechanism seems to help the network to focus on the important parts and get better results. In [Attention Is All You Need](https://arxiv.org/abs/1706.03762) the authors use only attention to perform the translation. Another example is [Attention-based LSTM Network for Cross-Lingual Sentiment Classification](http://www.aclweb.org/anthology/D/D16/D16-1024.pdf).
 
Hierarchical models have also been used for text classification, as in [HDLTex: Hierarchical Deep Learning for Text Classification](https://arxiv.org/abs/1709.08267) where HDLTex employs stacks of deep learning architectures to provide specialized understanding at each level of the document hierarchy.

In [Hierarchical Attention Networks (HAN) for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) the authors use the attention mechanism along with a hierarchical structure based on words and sentences to classify documents.

## Dataset

The dataset can be found in https://www.kaggle.com/c/msk-redefining-cancer-treatment/data. It contains basically the text of a paper, the gen related with the mutation and the variation. One text can have multiple genes and variations, so we will need to add this information to our models somehow.

One of the things we need to do first is to clean the text as it from papers and have a lot of references and things that are not relevant for the task. The second thing we can notice from the dataset is that the variations seem to follow some type of pattern. Although we might be wrong we will transform the variations in a sequence of symbols in order to let the algorithm discover this patterns in the symbols if it exists. We would get better results understanding better the variants and how to encode them correctly.

In the beginning of the kaggle competition the test set contained 5668 samples while the train set only 3321. The reason was most of the test samples were fake in order to not to extract any information from them. Later in the competition this test set was made public with its real classes and only contained 987 samples. We will use the test dataset of the competition as our validation dataset in the experiments. Every train sample is classified in one of the 9 classes, which are very unbalanced.
  
 |class|1|2|3|4|5|6|7|8|9|
 |-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
 |number of samples|566|452|89|686|242|273|952|19|37|
 
### Preprocessing
 
 Probably the most important task of this challenge is how to model the text in order to apply a classifier. As we don’t have deep understanding of the domain we are going to keep the transformation of the data as simple as possible and let the deep learning algorithm do all the hard work for us. But, most probably, the results would improve with a better model to extract features from the dataset.

We do several things to clean the data:
- Remove bibliographic references as “Author et al. 2007” or “[1,2]”. We also remove other paper related stuff like “Figure 3A” or “Table 4”.
- We add some extra white spaces around symbols as “.”, “,”, “?”, “(“, “0”, etc.
- We replace the numbers by symbols. If the number is below 0.001 is one symbol, if it is between 0.001 and 0.01 is another symbol, etc.
- We change all the variations we find in the text by a sequence of symbols where each symbol is a character of the variation (with some exceptions).

Another approach is to use a library like [nltk](http://www.nltk.org/) which handles most of the cases to split the text, although it won't delete things as the typical references to tables, figures or papers.
 
### Data augmentation
 
 Our dataset is very limited for a deep learning algorithm, we only count with 3322 training samples. In order to avoid overfitting we need to increase the size of the dataset and try to simplify the deep learning model.
 
 The best way to do data augmentation is to use humans to rephrase sentences, which it is an unrealistic approach in our case. Another way is to replace words or phrases with their synonyms, but we are in a very specific domain where most keywords are medical terms without synonyms, so we are not going to use this approach.
 
 As we have very long texts what we are going to do is to remove parts of the original text to create new training samples. We select a couple or random sentences of the text and remove them to create the new sample text.
 
 In order to improve the Word2Vec model and add some external information, we are going to use the definitions of the genes in the Wikipedia. Our hypothesis is that the external sources should contain more information about the genes and their mutations that are not in the abstracts of the dataset. We could add more external sources of information that can improve our Word2Vec model as others research papers related to the topic. We leave this for future improvements out of the scope of this article.

## Experiments

Like in the competition, we are going to use the [multi-class logarithmic loss](https://www.kaggle.com/wiki/LogLoss) for both training and test. 

If we would want to use any of the models in real life it would be interesting to analyze the [roc curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for all classes before taking any decision. In the scope of this article, we will also analyze briefly the accuracy of the models.

Next we are going to see the training set up for all models. The parameters were selected after some trials, we only show here the ones that worked better when training the models.

### Word2Vec

We use the Word2Vec model as the initial transformation of the words into embeddings for the rest of the models except the Doc2Vec model.

We use a linear context and [skip-gram with negative sampling](https://arxiv.org/abs/1301.3781), as it gets better results for small datasets with infrequent words. The context is generated by the 2 words adjacent to the target word and 2 random words of a set of words that are up to a distance 6 of the target word. Where the most infrequent words have more probability to be included in the context set. The vocabulary size is 40000 and the embedding size is 300 for all the models. We also use 64 negative examples to calculate the loss value.

We train the model for 2 epochs with a batch size of 128. The learning rate is 0.01 with a 0.9 decay every 100000 steps.

### Doc2Vec

We use a similar setup as in Word2Vec for the training phase. To prediction whether the doc vector belongs to one class or another we use 3 fully connected layers of sizes: 600, 300 and 75; with a dropout layer with a probability of 0.85 to keep the connection. All layers use a relu function as activation but the last one that uses softmax for the final probabilities. This prediction network is trained for 10000 epochs with a batch size of 128. The learning rate is 0.01 with 0.95 decay every 2000 steps.

### 3-Layer GRU

The first RNN model we are going to test is a basic RNN model with 3 layers of 200 GRU cells each layer

We train the model for 10 epochs with a batch size of 24 and a learning rate of 0.001 with 0.85 decay every 1000 steps. With these parameters some models we tested overfitted between epochs 11 and 15. These parameters are used in the rest of the deep learning models.

We use this model to test how the length of the sequences affect the performance. We test sequences with the first 1000, 2000, 3000, 5000 and 10000 words. We want to check whether adding the last part, what we think are the conclusions of the paper, makes any improvements, so we also tested this model with the first and last 3000 words. 

The output of the RNN network is concatenated with the embeddings of the gene and the variation. This concatenated layer is followed by a full connected layer with 128 hidden neurons and relu activation and another full connected layer with a softmax activation for the final prediction. In case of the model with the first and last words, both outputs are concatenated and used as input to the first fully connected layer along with the gene and variation. This set up is used for all the RNN models to make the final prediction, except in the ones we tell something different.

The combination of the first and last words got the best results as we will see below, and was the configuration used for the rest of the models.

The optimization algorithms is RMSprop with the default values in TensorFlow for all the next algorithms.

### Bidirectional GRU

This model only contains two layers of 200 GRU cells, one with the normal order of the words and the other with the reverse order. This is a bidirectional GRU model with 1 layer. 

### CNN + GRU

This model is 2 stacked CNN layers with 50 filters and a kernel size of 5 that process the sequence before feeding a one layer RNN with 200 GRU cells.

### HAN

This model is based in the model of [Hierarchical Attention Networks (HAN) for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) but we have replaced the context vector by the embeddings of the variation and the gene. This is, instead of learning the context vector as in the original model we provide the context information we already have.

As this model uses the gene and variation in the context vector of the attention we do not use the same full connected layer to make the predictions as in the other models. We use a simple full connected layer with a softmax activation function. A input we use a maximum of 150 sentences with 40 words per sentence (maximum 6000 words), gaps are filled with zeros.

## Results

### Competition results

As a baseline here we show some results of some competitors that made their kernel public. These are the kernels:
- [Bag of Words, TF-IDF, Word2Vec, LSTM](https://www.kaggle.com/reiinakano/basic-nlp-bag-of-words-tf-idf-word2vec-lstm)
- [XGBoost](https://www.kaggle.com/the1owl/redefining-treatment-0-57456)

The results of those algorithms are shown in the next table. In the case of this experiments, the validation set was selected from the initial training set.

| Algorithm | Validation Loss | Validation Accuracy| Public Leaderboard Loss |
|-|:-:|:-:|:-:|
| Bag of words          | 1.65 | 48% | - |
| Random Forest         | 1.44 | 50% | - |
| TF-IDF + SVC          | 1.20 | 55% | - |
| Word2Vec + XGBoost    | 1.26 | 54% | 0.96 |
| LSTM                  | 1.44 | 48% | 1.00 |
| XGBoost               | 1.06 | - | 0.57 |

In general, the public leaderboard of the competition shows better results than the validation score in their test. This could be due to a bias in the dataset of the public leaderboard. A different distribution of the classes in the dataset could explain this bias but as I analyzed this dataset when it was published I saw the distribution of the classes was similar.

Analyzing the algorithms the deep learning model based on LSTM cells doesn't seem to get good results compared to the other algorithms. But as one of the authors of those results explained, the LSTM model seems to have a better distributed confusion matrix compared with the other algorithms. He concludes it was worth to keep analyzing the LSTM model and use longer sequences in order to get better results. We will see later in other experiments that longer sequences didn't lead to better results.

### Deep learning models

First, we wanted to analyze how the length of the text affected the loss of the models with a simple 3-layer GRU network with 200 hidden neurons per layer. We also checked whether adding the last part, what we think are the conclusions of the paper, makes any improvements. We added the steps per second in order to compare the speed the algorithms were training. We used 3 GPUs Nvidia k80 for training. The results are in the next table:

| Algorithm | Validation Loss | Validation Accuracy | Steps per second |
|-|:-:|:-:|:-:|
| First 1000 words                      | 1.61 | 37.2% | 0.97 |
| First 2000 words                      | 1.91 | 29.6% | 0.47 |
| First 3000 words                      | 1.94 | 28.8% | 0.31 |
| First 5000 words                      | 1.86 | 29.9% | 0.19 |
| First 10000 words                     | 1.70 | 41.9% | 0.09 |
| First 3000 words + Last 3000 words    | 1.95 | 29.4% | 0.14 |

Results are very similar for all cases, but the experiment with less words gets the best loss while the experiment with more words gets the best accuracy in the validation set. In all cases the number of steps per second is inversely proportional to the number of words in the input. More words require more time per step.

To compare different models we decided to use the model with 3000 words that used also the last words. This is the biggest model that fit in memory in our GPUs. These are the results:

| Algorithm | Validation Loss | Validation Accuracy | Steps per second |
|-|:-:|:-:|:-:|
| Doc2Vec                           | 1.56 | 47.9% | - |
| 3-layer GRU                       | 1.95 | 29.4% | 0.31 |
| Bidirectional GRU                 | 2.08 | 20.4% | 0.21 |
| CNN + GRU                         | 1.99 | 23.2% | 0.15 |
| HAN + gene-variation context      | 1.60 | 41.0% | 1.58 |

It seems that the bidirectional model and the CNN model perform very similar to the base model. The HAN model seems to get the best results with a good loss and goo accuracy, although the Doc2Vec model outperforms this numbers. The HAN model is much faster than the other models due to use shorter sequences for the GRU layers.

In the next image we show how the embeddings of the documents in doc2vec are mapped into a 3d space where each class is represented by a different color. We don't appreciate any clear aggrupation of the classes, regardless it was the best algorithm in our tests:

![Doc2Vec doc embeddings classes](https://raw.githubusercontent.com/jorgemf/kaggle_redefining_cancer_treatment/master/img/doc2vec_doc_embeddings.png "Doc2Vec doc embeddings classes")

### Jupyter notebook

Similar to the previous model but with a different way to apply the attention we created a kernel in kaggle for the competition: [RNN + GRU + bidirectional + Attentional context](https://www.kaggle.com/jorgemf/rnn-gru-bidirectional-attentional-context). The network was trained for 4 epochs with the training and validation sets and submitted the results to kaggle. I used both the training and validation sets in order to increase the final training set and get better results. The 4 epochs were chosen because in previous experiments the model was overfitting after the 4th epoch. It scored 0.93 in the public leaderboard and 2.8 in the private leaderboard. Public leaderboard was usually 0.5 points better in the loss compared to the validation set.

The confusion matrix shows a relation between the classes 1 and 4 and also between the classes 2 and 7. The classes 3, 8 and 9 have so few examples in the datasets (less than 100 in the training set) that the model didn't learn them.

![Confusion matrix](https://raw.githubusercontent.com/jorgemf/kaggle_redefining_cancer_treatment/master/img/confusion_matrix.png "Confusion matrix")

## Discussion

Giver all the results we observe that non-deep learning models perform better than deep learning models. It could be to the problem of RNN to generalize with long sequences and the ability of non-deep learning methods to extract more relevant information regardless of the text length. The hierarchical model may get better results than other deep learning models because of its structure in hierarchical layers that might be able to extract better information.

The number of examples for training are not enough for deep learning models and the noise in the data might be making the algorithms to overfit to the training set and to not extract the right information among all the noise.

## Conclusions

The kaggle competition had 2 stages due to the initial test set was made public and it made the competition irrelevant as anyone could submit the perfect predictions. That is why the initial test set was made public and a new set was created with the papers published during the last 2 months of the competition. This leads to a smaller dataset for test, around 150 samples, that needed to be distributed between the public and the private leaderboard. When the private leaderboard was made public all the models got really bad results. Almost all models increased the loss around 1.5-2 points. The huge increase in the loss means two things. First, the new test dataset contained new information that the algorithms didn't learn with the training dataset and couldn't make correct predictions. This is normal as new papers try novelty approaches to problems, so it is almost completely impossible for an algorithm to predict this novelty approaches. Second, the training dataset was small and contained a huge amount of text per sample, so it was easy to overfit the models.

Regardless the deep learning model shows worse results in the validation set, the new test set in the competition proved that the text classification for papers is a very difficult task and that even good models with the currently available data could be completely useless with new data. As the research evolves, researchers take new approaches to address problems which cannot be predicted. With a bigger sample of papers we might create better classifiers for this type of problems and this is something worth to explore in the future. These new classifiers might be able to find common data in the research that might be useful, not only to classify papers, but also to lead new research approaches.

## Appendix: How to reproduce the experiments in TensorPort

This project requires Python 2 to be executed.

### Initial set up

Let's install and login in [TensorPort](https://tensorport.com) first:

```sh
pip install --upgrade tensorport
pip install --upgrade git-lfs 
tport login
```

Now set up the directory of the project in a environment variable. We also set up other variables we will use later. You need to set up the correct values here:

```sh
export PROJECT=kaggle-redefining-cancer-treatment
export DATASET=kaggle-redefining-cancer-treatment-data
export PROJECT_DIR=/home/jorge/projects/$PROJECT
export TPORT_USER=jorgemf
```

Clone the repo and install the dependencies for the project:

```sh
git clone git@github.com:jorgemf/kaggle_redefining_cancer_treatment.git $PROJECT_DIR
cd $PROJECT_DIR
pip install -r requirements.txt
```

Change the dataset repository, you have to modify the variable `DIR_GENERATED_DATA` in `src/configuration.py`. Change `$TPORT_USER` and `$DATASET` by the values set before.

```python
DIR_GENERATED_DATA = get_data_path(dataset_name='$TPORT_USER/$DATASET',
                                   local_root=DIR_DATA,
                                   local_repo='generated')
```

Commit the changes:

```sh
git commit -m "Updated the data repository"
```

### Process the data

```sh
mkdir -p $PROJECT_DIR/data/generated
```

You first need to download the data into the `$PROJECT_DIR/data` directory from the [kaggle competition page](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data). Unzip the data in the same directory.

Now let's process the data and generate the datasets. This takes a while.

```sh
python -m src.preprocess_data
python -m src.w2v.word2vec_process_data
python -m src.rnn.text_classification_process_data
python -m src.d2v.doc2vec_process_data
python -m src.preprocess_data_stage2
```

### Word2Vec

We need the word2vec embeddings for most of the experiments. In this case we run it locally as it doesn't require too many resources and can finish in some hours.

```sh
python -m src.w2v.word2vec_train
```

### Upload repo and data to TensorPort servers

We need to upload the data and the project to TensorPort in order to use the platform. We use `$PROJECT` as the name for the project and dataset in TensorPort.

```sh
tport create project --name $PROJECT
git push tensorport master
cd data/generated/
git init
git-lfs track word2vec/* text_classification/*
git add .gitattributes word2vec/* text_classification/* 
git commit -m "update data"
tport create dataset --name $DATASET
git push tensorport master
cd $PROJECT_DIR
```

Note as not all the data is uploaded, only the generated in the previous steps for word2vec and text classification. Doc2vec is only run locally in the computer while the deep neural networks are run in TensorPort.

### 3-layer GRU with first and last words experiments

#### First 1000 words

In `src/configuration.py` set these values:

```python 
MAX_WORDS = 1000 
USE_END_SEQUENCE = False 
```

and commit the changes:

```sh
git commit -m "MAX_WORDS=1000 and USE_END_SEQUENCE=False"
git push tensorpot master
```

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "simple-1000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_simple" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""

#### First 2000 words

In `src/configuration.py` set these values:

```python 
MAX_WORDS = 2000 
USE_END_SEQUENCE = False 
```

and commit the changes:

```sh
git commit -m "MAX_WORDS=2000 and USE_END_SEQUENCE=False"
git push tensorpot master
``` 

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "simple-2000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_simple" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

We could use 4 ps replicas with the basic plan in TensorPort but with 3 the data is better distributed among them. With 4 ps replicas 2 of them have very small data.

#### First 3000 words

In `src/configuration.py` set these values:

```python 
MAX_WORDS = 3000 
USE_END_SEQUENCE = False 
```

and commit the changes:

```sh
git commit -m "MAX_WORDS=3000 and USE_END_SEQUENCE=False"
git push tensorpot master
``` 

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "simple-3000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_simple" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

#### First 5000 words

In `src/configuration.py` set these values:

```python 
MAX_WORDS = 5000 
USE_END_SEQUENCE = False 
```

and commit the changes:

```sh
git commit -m "MAX_WORDS=5000 and USE_END_SEQUENCE=False"
git push tensorpot master
``` 

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "simple-5000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_simple" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

#### First 10000 words

In `src/configuration.py` set these values:

```python 
MAX_WORDS = 10000 
USE_END_SEQUENCE = False 
```

and commit the changes:

```sh
git commit -m "MAX_WORDS=10000 and USE_END_SEQUENCE=False"
git push tensorpot master
```  

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "simple-10000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_simple" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

#### First 3000 words + Last 3000 words  

In `src/configuration.py` set these values:

```python 
MAX_WORDS = 3000 
USE_END_SEQUENCE = True 
```

and commit the changes:

```sh
git commit -m "MAX_WORDS=3000 and USE_END_SEQUENCE=True"
git push tensorpot master
```  

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "simple-with_end-3000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_simple" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

We will use this configuration for the rest of the models executed in TensorPort.

### Experiments with different models

#### Doc2Vec               

We also run this experiment locally as it requires similar resources as Word2Vec.

First, we generate the embeddings for the training set:

```sh
python -m src.d2v.doc2vec_train_word_embeds
```

Second, we generated the model to predict the class given the doc embedding:

```sh
python -m src.d2v.doc2vec_train_doc_prediction
```

Third, we generate the doc embeddings for the evaluation set:

```sh
python -m src.d2v.doc2vec_train_eval_word_embeds train_val
```

Finally, we evaluate the doc embeddings with the predictor of the second step:

```sh
python -m src.d2v.doc2vec_eval_doc_prediction val
```

#### Bidirectional GRU

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "bidirectional-gru-with_end-3000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_bidirectional" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

#### CNN + GRU

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "cnn-with_end-3000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_cnn" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```

#### HAN

Launch a job in TensorPort. You have to select the last commit (number 0). The last worker is used for validation, you can check the results in the logs.

```sh
tport run --name "han-with_end-3000" --project "$TPORT_USER/$PROJECT:$(git rev-parse HEAD)" --datasets "$TPORT_USER/$DATASET:$(cd data/generated/ && git rev-parse HEAD)" --package-path "src.rnn" --module "text_classification_model_han" --python-version 2 --tf-version "1.2.0" --requirements "requirements.txt" --distributed --worker-replicas 4 --worker-type "p2.xlarge" --ps-replicas 3 --ps-type "t2.small" --time-limit "24h00m" --description ""
```
