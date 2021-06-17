# Laboro BERT Japanese: Japanese BERT Pre-Trained With Web-Corpus

<!-- TOC -->

- [Introduction](#introduction)
    - [About this model](#about-this-model)
    - [How well is the performance](#how-well-is-the-performance)
    - [To cite this work](#to-cite-this-work)
    - [License](#license)
- [Fine-Tuning with Our Model](#fine-tuning-with-our-model)
    - [Classification](#classification)
        - [Dataset - Livedoor News Corpus](#dataset---livedoor-news-corpus)
        - [Requirements](#req1)
        - [To use the code](#code1)
    - [Question Answering](#question-answering)
        - [Dataset - Driving Domain QA](#dataset---driving-domain-qa)
        - [Requirements](#req2)
        - [To use the code](#code2)
- [About the Pre-Training of Our Model](#about-the-pre-training-of-our-model)
    - [Corpus](#corpus)
    - [SentencePiece Model](#sentencepiece-model)
    - [Pre-Training](#pre-training)
        - [Hyper-parameters](#hyper-parameters)
        - [Environment](#environment)

<!-- /TOC -->

## Introduction

### About our BERT model

This Japanese BERT model was pre-trained with our own web corpus, on the basis of the [original BERT](https://github.com/google-research/bert) and this [Japanese BERT](https://github.com/yoheikikuta/bert-japanese). So far both base model (12-layer, 768-hidden, 12-heads, 110M parameters) and large model (24-layer, 1024-hidden, 16-heads, 340M parameters) pre-trained with the same web corpus have been released.

   Download the base model from [here](http://assets.laboro.ai.s3.amazonaws.com/laborobert/webcorpus_base_model.zip).  
   Download the large model from [here](http://assets.laboro.ai.s3.amazonaws.com/laborobert/webcorpus_large_model.zip).  

### How well is the performance

The models have been evaluated for two tasks, Livedoor news classification task and driving-domain question answering (DDQA) task. In Livedoor news classification, each piece of news is supposed to be classified into one of nine categories. In DDQA task, given question-article pairs, answers to the questions are expected to be found from the articles. The results of the evaluation are shown below, in comparison with a baseline model pre-trained with Japanese Wikipedia corpus released by this [Japanese BERT](https://github.com/yoheikikuta/bert-japanese) repository. Note that the results are the averages of multiple-time mearsurement. Due to the small size of the evaluation datasets, the results may differ a little every time.

For Livedoow news classification task:

| model size | corpus | corpus size | eval evironment | batch size | epoch | learning rate | measurement times | mean accuracy (%) | standard deviation |
|-|-|-|-|-|-|-|-|-|-|
| Base | JA-Wikipedia | 2.9G | GPU | 4  | 10 | 2e-5 | 5  | 97.23 | 2.38e-1 |
| Base | Web Corpus   | 12G  | GPU | 4  | 10 | 2e-5 | 5  | 97.72 | 2.27e-1 |
| Large | Web Corpus  | 12G  | TPU | 32 | 7  | 2e-5 | 30 | 98.07 | 2.45e-3 |

For Driving-domain QA task:

| model size | corpus | corpus size | eval evironment | batch size | epoch | learning rate | measurement times | mean EM (%) | standard deviation |
|-|-|-|-|-|-|-|-|-|-|
| Base | JA-Wikipedia | 2.9G | TPU | 32 | 3 | 5e-5 | 100 | 76.3 | 5.16e-3 |
| Base | Web Corpus   | 12G  | TPU | 32 | 3 | 5e-5 | 100 | 75.5 | 5.06e-3 |
| Large | Web Corpus  | 12G  | TPU | 32 | 3 | 5e-5 | 30  | 77.3 | 4.96e-3 |

### To cite this work
We haven't published any paper on this work.
Please cite this repository:
```
@article{Laboro BERT Japanese,
  title = {Laboro BERT Japanese: Japanese BERT Pre-Trained With Web-Corpus},
  author = {"Zhao, Xinyi and Hamamoto, Masafumi and Fujihara, Hiromasa"},
  year = {2020},
  howpublished = {\url{https://github.com/laboroai/Laboro-BERT-Japanese}}
}
```

### License
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a></br>

   This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.  
   For commercial use, please [contact Laboro.AI Inc.](https://laboro.ai/contact/other/)

## Fine-Tuning with Our Model

### Classification

Text classification means assigning labels to text. Because the labels can be defined to describe any aspect of the text, text classification has a wide range of application. The most straightforward one would be categorizing the topic or sentiment of the text. Besides those, other examples include recognizing spam email, judging whether two sentences have same or similar meaning.

#### Dataset - Livedoor News Corpus

In the evaluation of English BERT model in classification task, several datasets (e.g. SST-2, MRPC) can be used as common benchmarks. As for Japanese BERT model, [Livedoor news corpus](https://www.rondhuit.com/download.html#ldcc) can be used in the same fashion. Each piece of news in this corpus can be classified into one of the nine categories.

The original corpus is not devided in training, evaluation, and testing data. The dataset we provided in this repository was pre-processed based on Livedoor News Corpus in following steps: 
* concatenating all of the data
* shuffling randomly
* deviding into train:dev:test = 2:2:6

<a name="req1"></a>

#### Requirements

* Python 3.6.9
* tensorflow==1.13.0
* sentencepiece==0.1.85
* GPU is recommended

<a name="code1"></a>

#### To use the code

Before running the code, make sure

* the livedoor dataset is in the data folder
* the pre-trained BERT model is in the model folder, including model.ckpt.data, model.ckpt.meta, model.ckpt.index, bert_config.json
* the sentencepiece model is also in the model folder, including webcorpus.model, webcorpus.vocab

```bash
git clone https://github.com/laboroai/Laboro-BERT-Japanese.git
cd ./Laboro-BERT-Japanese/src
./run_classifier.sh
```

### Question Answering

Question answering task is another way to evaluate and apply BERT model. In English NLP, [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) is one the of most widely used datasets for this task. In SQuAD, questions and corresponding Wikipedia pages are given, and the answers to the questions are supposed to be found from the Wikipedia pages.

#### Dataset - Driving Domain QA

For QA task, we used [Driving Domain QA dataset](http://nlp.ist.i.kyoto-u.ac.jp/index.php?Driving%20domain%20QA%20datasets) for evaluation. This dataset consists of PAS-QA dataset and RC-QA dataset. So far, we have only evaluated our model on the RC-QA dataset. The dataset is already in the format of SQuAD 2.0, so no pre-processing is needed for further use.

<a name="req2"></a>

#### Requirements

* Python 3.6.9
* tensorflow==1.13.0
* sentencepiece==0.1.85
* TPU is recommended (in our experiments, out-of-memory error occurs when using GPU)
* Google Cloud Storage if TPU is used

<a name="code2"></a>

#### To use the code

TPU is recommended for this evaluation, and [TPU can only read from and write to Google Cloud Storage](https://cloud.google.com/tpu/docs/troubleshooting#cannot_use_local_filesystem), thus we recommend to place BERT model and output in cloud storage bucket. Before running the code, make sure

* the livedoor dataset is in the data folder
* the pre-trained BERT model is in the model folder in **cloud storage bucket**, including model.ckpt.data, model.ckpt.meta, model.ckpt.index, bert_config.json
* the sentencepiece model is in the **local** model folder, including webcorpus.model, webcorpus.vocab

```bash
git clone https://github.com/laboroai/Laboro-BERT-Japanese.git
cd ./Laboro-BERT-Japanese/src
./run_squad.sh
```

## About the Pre-Training of Our Model

### Corpus

Our Japanese BERT model is pre-trained with a web-based corpus especially built for this project. It was built by using a web crawler, and in total 2,605,280 webpages from 4,307 websites were crawled. The source websites extend from news websites and part of Wikipedia to personal blogs, covering both formal and informal written Japanese.

The original English BERT model was trained on a 13GB corpus consisting of English Wikipedia and BooksCorpus. The size of raw text in our web-based corpus is 12GB, which is similar to the original one.

### SentencePiece Model

[SentencePiece](https://github.com/google/sentencepiece) is used as the tokenizer. The parameters when training the sentencepiece model are as followings:

```python
vocab_size = 32000
shuffle_input_sentence = True
input_sentence_size = 18000000
character_coverage = 0.9995 #default
model_type = 'unigram' #default
ctlsymbols = '[CLS],[SEP],[MASK]'
```

### Pre-Training

#### Hyper-parameters

The pre-training consists of two phases, in which the ```train_batch_size``` and ```max_sequence_length``` are changed.

Phase 1
```python
train_batch_size = 256
max_seq_length = 128
num_train_steps = 2900000
num_warmup_steps = 10000
learning_rate = 1e-4
```

Phase 2
```python
train_batch_size = 64
max_seq_length = 512
num_train_steps = 3900000
num_warmup_steps = 10000
learning_rate = 1e-4
```

#### Environment

* [Cloud TPU](https://cloud.google.com/tpu/) v3-8 on Google Cloud Platform
* tensorflow==1.13.0




