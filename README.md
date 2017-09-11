[Read this in medium](https://medium.com/@jorgemf/personalized-medicine-redefining-cancer-treatment-by-deep-learning-f6c64a366fff)

```bash 
python src/preprocess_data.py
python src/word2vec_process_data.py
python src/text_classification_process_data.py
python src/doc2vec_process_data.py
```

doc2vec

```bash
python src/doc2vec_train_word_embeds.py
python src/doc2vec_train_doc_prediction.py
python src/doc2vec_eval_doc.py
```

word2vec

```bash
python src/word2vec_train.py
```

simple model multilayer

```bash
python src/text_classification_model_simple.py 
python src/text_classification_model_simple.py eval
```

simple model + cnn

```bash
python src/text_classification_model_simple_cnn.py 
python src/text_classification_model_simple_cnn.py eval
```

simple model bidirectional

```bash
python src/text_classification_model_simple_bidirectional.py 
python src/text_classification_model_simple_bidirectional.py eval
```

hatt

```bash
python src/text_classification_model_hatt.py 
python src/text_classification_model_hatt.py eval
```

## TensorPort

```bash
tport login
```

```bash 
tport create job --name "word2vev-distributed-8" --project "jorgemf/kaggle-personalized-medicine-redefining-cancer-treatment-by-deep-learning" --datasets "jorgemf/kaggle-redefining-cancere-treatment"  --module "word2vec_train" --package-path "src" --python-version 3 --tf-version "1.2" --requirements "requirements.txt" --single-node --instance-type "t2.small" --time-limit "07h00m"

tport create job --name "word2vev-distributed-8" --project "jorgemf/kaggle-personalized-medicine-redefining-cancer-treatment-by-deep-learning" --datasets "jorgemf/kaggle-redefining-cancere-treatment"  --module "word2vec_train" --package-path "src" --python-version 3 --tf-version "1.2" --requirements "requirements.txt" --worker-replicas 3 --worker-type "p2.xlarge" --ps-replicas 1 --ps-type "c4.2xlarge"	 --time-limit "06h00m"

tport run --job-id "word2vev-distributed-8"
```