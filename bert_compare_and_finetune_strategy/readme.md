# The Implementation for Getting the Best Out of a BERT Model

## Introduction

To fine-tune a BERT model means to adapt a general-purpose model to one specific down-stream NLP task. Any changes in the tokenizer and hyper-parameteres can influence the performance of the fine-tuned model. We compared 10 Japanese BERT models that are pre-trained using various pre-training hyper-parameters, implementations, and training corpora. We evaluated them all on three tasks and managed to summarize the fine-tuning strategy in order to take full advantage of the models and optimize their performance. We published [an article](∆) to introduce in details our strategy. It is worth reading if you have a pre-trained BERT model and want to get the most out of it.

This repository contains the scripts to utilize our fine-tuning strategy based on Transformers, and you can find the instruction for using the scripts in this document. We also include the comparison results of the 10 Japanese BERT models as the last section in this document.

<!-- TOC -->

-   [Introduction](#introduction)
    -   [To cite this work](#to-cite-this-work)
    -   [License](#license)
-   [Requirements](#requirements)
-   [Implementation](#implementation)
    -   [Convert Checkpoints](#convert-tensorflow-checkpoints-to-pytorch-checkpoints)
    -   [Preprocess](#preprocess-datasets)
    -   [Fine-tune & Evaluate](#fine-tune-&-evaluate)
-   [Results](#results)
    -   [Best Performance](#best-performance)
    -   [Unigram vs BPE](#unigram-vs-bpe)
    -   [Tokenizer Configuration](#tokenizer-configuration)
    -   [Learning Rate](#learning-rate)
    -   [Max Sequence Length](#max-sequence-length)
    -   [Unanswerable Questions Threshold](#unanswerable-questions-threshold)

<!-- /TOC -->

## Requirements

    mecab-python3==1.0.3
    mojimoji==0.0.11
    pyknp==0.4.6
    sentencepiece==0.1.95
    torch==1.8.0
    transformers==2.4.1

## Implementation

Our implementation combines the [NICT instruction](https://alaginrc.nict.go.jp/nict-bert/Experiments_on_RCQA.html#BERT-%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89) and our strategy to improve the performance of a pre-trained Japanese BERT model in the fine-tuning phase. It's done in 3 steps,
* convert checkpoints
* preprocess
* fine-tune & evaluate

### Convert Tensorflow Checkpoints to PyTorch Checkpoints

A complete Tensorflow checkpoint should include all the following files. The number in the file name stands for how many steps the model has been trained for. In our example below, the model was trained for 1,400,000 steps.

```txt
├── config.json
├── model.ckpt-1400000.data-00000-of-00001
├── model.ckpt-1400000.index
└── model.ckpt-1400000.meta
```

To use Transformers for fine-tuning and evaluating BERT models, the TF checkpoints have to be converted into PyTorch checkpoints first. The script below can convert the checkpoint and save the output in the same directory as the original checkpoint.

```bash
bash src/convert_model.sh [path-to-checkpoint]/model.ckpt-1400000
```

### Preprocess Datasets

All models are tested on 3 tasks. LDCC is a news classification task. DDQA is a QA task in SQuAD v1.1 format, which means all the questions are answerable. In contrast, RCQA is a QA task in SQuAD v2.0 format. In order to better simulate the actual situation when you browse an article trying to find an answer but couldn't, SQuAD v2.0 added questions that cannot be answered.

Please follow the instructions and download the datasets, and then place them in `data` folder.
* [LDCC](https://www.rondhuit.com/download.html#ldcc)
* [DDQA](https://nlp.ist.i.kyoto-u.ac.jp/?Driving+domain+QA+datasets)
* [RCQA](http://www.cl.ecei.tohoku.ac.jp/rcqa/)

```
data
├── ddqa
│   └── DDQA-1.0
├── ldcc
│   └── text
└── rcqa
    └── all-v1.0.json.gz
```

We provide the Python scripts to preprocess the datasets. You can directly run the scripts as shown below to preprocess for all BERT models. The scripts also support preprocessing datasets for an individual model, and you can specify the tokenizer by passing `--mode` and `--expe` arguments.

```bash
python3 src/preprocess_ldcc.py
python3 src/preprocess_ddqa.py
python3 src/preprocess_rcqa.py
```

### Fine-tune & Evaluate

In [another article](∆), we introduced in details our strategy to optimize the performance of a BERT model in fine-tuning phase. It is worth reading if you have a pre-trained BERT model and want to get the most out of it. Our strategy focuses on the following aspects,

* Tokenizer
    * unigram vs BPE
    * correctly use the Transformers built-in tokenizers
* Fine-Tuning Hyper-parameters
    * those you should test every possibility
    * those that should fit your model
    * threshold for unanswerable QA task

Accordingly, based on the Transformers example fine-tuning scripts `run_glue.py` and `run_qa.py` (previously as `run_squad.py`), we re-wrote part of the scripts to make it more convenient to adopt our strategy. In the rest of this section, you'll read 1) how to use our strategy in an actual implementation, 2) the scripts we provide and the arguments explanation.

Whether to use unigram or BPE tokenizer was pre-decided in pre-training, and it can influence how to utilize the Transformers built-in tokenizers in fine-tuning. The tokenization for models based on unigram tokenizers has finished in the pre-processing step, and no further tokenization is needed after that. So during fine-tuning, we used `BertJapaneseTokenizer` and turn off the sub-word tokenization. On the other hand, models based on BPE tokenizer need two-step tokenization. In the pre-processing, MeCab was used to do the word tokenization. In the fine-tuning step, `BertTokenizer` should be used for the sub-word tokenization. 

In the tokenizer configuration, `use_lower_case` should always be set as false, otherwise the Dakuten ( ﾞ) and Handakuten ( ﾟ) will disappear. Also, `tokenize_chinese_chars` should be set as false, so that the tokenizer wouldn't add redundant whitespace around Chinese characters. The easy way to make sure the tokenizer is correctly configured is to place a `tokenizer_config.json` file under the directory of your model. We provide a sample configuration file in `model/sample_config/tokenizer_config.json`.

The hyper-parameters, such as batch size, learning rate, training epochs, and max sequence length, can be adjusted by passing the corresponding arguments when running the script. Among those, the batch size might be a bit tricky to understand. It equals `number of GPU` * `per_gpu_train_batch_size` * `gradient_accumulation_steps`, in which the number of GPU is different between cases. The `per_gpu_train_batch_size` is usually set as `8` because it might cause an OOM problem if the number is any bigger. And the gradient will be accumulated over `gradient_accumulation_steps` steps.

As for the QA task with unanswerable questions, we found that setting the `null_score_diff_threshold` can improve the performance. The optimal threshold can be automatically calculated during the evaluation on the dev set. All you have to do is to run `src/run_squad_rcqa_args.py` and then `src/run_squad_rcqa_only_eval_args.py`. 


You can find the following scripts for fine-tuning.
* `src/run_glue_args.py` for LDCC text classification task
* `src/run_squad_ddqa_args.py` for DDQA question answering task
* `src/run_squad_rcqa_args.py` and `src/run_squad_rcqa_only_eval_args.py` for RCQA question answering task

All the scripts share the following arguments.

```
data_dir                        the directory name for the tokenized data
model_name_or_path              the directory name for the model
prefix                          the prefix to name the fine-tuned model; it usually contains information like max_seq_length and which tokenizer was used
--root_data_dir                 the root directory to find data_dir; default: "/home/ubuntu/bert_compare/data/[task-name]"
--root_model_name_or_path       the root directory to find the model directory
--tokenizer_type                1-BertTokenizer 2-BertJapaneseTokenizer; type: int; default: 1 
--max_seq_length                max sequence length; default: 512
--per_gpu_train_batch_size      batch size per GPU for training; default: 16
--gradient_accumulation_steps   the gradient will be accumulated over gradient_accumulation_steps steps; default: 1
--learning_rate                 the learning rate; default: 2e-5
--num_train_epochs              the number of training epochs; default: 3
--save_steps                    the number of interval steps to save a checkpoint, the default is set as a large number so that no middle checkpoint will be saved; default: 20000
```


## Results

### Best Performance

The table below shows the best performance we obtained for each model.
* For LDCC task, LaboroBERT-base-BPE model gives the best performance, followed by the other three LaboroBERT models, and NICT-BPE model is barely as good as LaboroBERT-base-unigram model.
* While for the other two QA tasks, NICT-BPE model has the highest score, followed by NICT-noBPE and UKyoto-large model. Laboro-large-BPE model also performs well on DDQA task.

| MODEL | LDCC ACC | LDCC F1 | RCQA EM | RCQA F1 | DDQA EM | DDQA F1 |
|-|-|-|-|-|-|-|
| Laboro-large-unigram | 97.28 | 97.22 | 73.66 | 76.63 | 76.68 | 86.24 |
| Laboro-large-BPE | 97.01 | 96.98 | 73.54 | 75.25 | 86.72 | 92.03 |
| Laboro-base-unigram | 96.74 | 96.67 | 72.86 | 76.00 | 72.85 | 84.34 |
| Laboro-base-BPE | 97.55 | 97.60 | 73.26 | 74.79 | 86.42 | 91.19 |
| UKyoto-large | 93.75 | 93.67 | 75.67 | 77.25 | 87.30 | 91.58 |
| UKyoto-base | 90.76 | 90.57 | 72.87 | 74.44 | 84.77 | 89.84 |
| NICT-BPE | 96.74 | 96.66 | 78.44 | 79.76 | 89.36 | 93.28 |
| NICT-noBPE | 96.33 | 96.18 | 76.74 | 78.20 | 87.21 | 91.58 |
| UTohoku | 96.06 | 95.88 | 75.67 | 75.97 | 87.11 | 91.00 |
| bert-wiki-ja | 96.60 | 96.36 | 73.79 | 76.96 | 74.70 | 85.23 |

### Unigram vs BPE

 Curious to see which tokenizer works better on which task, we pre-trained two models with the exact same parameters and corpus but different tokenizers.

| MODEL | LDCC ACC | LDCC F1 | RCQA EM | RCQA F1 | DDQA EM | DDQA F1 |
|-|-|-|-|-|-|-|
| Laboro-large-unigram | 97.28 | 97.22 | 73.66 | 76.63 | 76.68 | 86.24 |
| Laboro-large-BPE | 97.01 | 96.98 | 73.54 | 75.25 | 86.72 | 92.03 |

From the comparison, we can see that
* The model using unigram tokenizer performs better on sentence classification (LDCC) task
* The model using BPE tokenizer perform better on DDQA task. It makes sense because with word and subword tokenization, BPE tokenizer tends to split a sentence into finer pieces. By doing this, the searching for starting and ending indices for the answer can be more accurate.
* For QA task with unanswerable questions as in SQuAD v2.0, unigram and BPE give very similar performance. This is probably because SQuAD v2.0 is a combination of classification and QA tasks instead of a simple QA task.
* For other models based on Japanese Wikipedia data, although they are not pre-trained with the exact same hyper-parameters, the same conclusions can be made when comparing bert-wiki-ja model using unigram tokenizer to other models using BPE tokenizers.

### Tokenizer Configuration

To correctly configure the tokenizer, both `tokenize_chinese_chars` and `use_lower_case` should be false. To show how much harm mistakenly configuring a tokenizer can do, the following table compares when `tokenize_chinese_chars` is correctly set as False (TCC=False) and wrongly set as True (TCC=True). In all of our experiments, the drop always happens no matter which model is evaluated on which task.

| MODEL | LDCC ACC | LDCC F1 | RCQA EM | RCQA F1 | DDQA EM | DDQA F1 |
|-|-|-|-|-|-|-|
| Laboro-large-BPE, TCC=False | 97.01 | 96.98 | 73.54 | 75.25 | 86.72 | 92.03 |
| Laboro-large-BPE, TCC=True | 96.47 | 96.44 | 72.05 | 73.83 | 85.45 | 90.88 |
| Laboro-base-BPE, TCC=False | 97.55 | 97.60 | 73.26 | 74.79 | 86.42 | 91.19 |
| Laboro-base-BPE, TCC=True | 95.38 | 95.44 | 71.54 | 73.23 | 84.86 | 90.49 |
| UKyoto-large, TCC=False | 94.70 | 94.56 | 75.87 | 77.44 | 86.72 | 91.47 |
| UKyoto-large, TCC=True | 92.93 | 92.77 | 73.15 | 74.98 | 83.98 | 89.11 |
| UKyoto-base, TCC=False | 94.43 | 94.26 | 73.36 | 75.02 | 84.47 | 89.80 |
| UKyoto-base, TCC=True | 93.21 | 93.14 | 71.03 | 72.85 | 80.57 | 86.16 |

### Learning Rate

As mentioned in Appendix A.3, Devlin et al., 2019, there are 3 hyper-parameters we want to tune and their optimized values fall in these ranges for all tasks most of the time:

* batch size: 16, 32
* learning rate: 5e-5, 3e-5, 2e-5
* number of training epochs: 2, 3, 4

The table below shows the accuracy and F1 scores for LDCC task when different learning rates were used for fine-tuning. Unfortunately, no obvious pattern that suits every model can be found. As a result, we suggest trying every possible batch size, learning rate, and training epochs for you model.

| MODEL | LDCC EM, lr=2e-5 | LDCC F1, lr=2e-5 | LDCC ACC, lr=3e-5 | LDCC F1, lr=3e-5 | LDCC EM, lr=5e-5 | LDCC F1, lr=5e-5 |
|-|-|-|-|-|-|-|
| Laboro-large-unigram | 97.28 | 97.22 | 97.15 | 97.06 | 96.60 | 96.49 | 
| Laboro-large-bpe | 96.88 | 96.75 | 97.01 | 96.93 | 97.01 | 96.98 |
| Laboro-base-unigram | 94.97 | 94.97 | 96.74 | 96.67 | 96.66 | 96.50 |
| Laboro-base-bpe | 96.74 | 96.70 | 97.55 | 97.60 | 97.28 | 97.30 |
| UKyoto-large | 91.71 | 91.60 | 92.39 | 92.20 | 93.75 | 93.67 |
| UKyoto-base | 90.22 | 90.10 | 90.35 | 90.30 | 90.76 | 90.57 |
| NICT-BPE | 95.38 | 95.22 | 96.74 | 96.66 | 94.84 | 94.58 |
| NICT-noBPE | 95.79 | 95.66 | 96.33 | 96.18 | 95.92 | 95.76 |
| UTohoku | 95.24 | 95.10 | 95.11 | 95.00 | 96.06 | 95.88 |
| bert-wiki-ja | 95.65 | 95.50 | 96.06 | 95.94 | 96.60 | 96.36 |

### Max Sequence Length

While it is suggested using the same max sequence length in fine-tuning as in pre-training, we tried shorter length for some models on the 3 tasks to see how much worse the performance will become. Specifically, we used 128 as the shorter length for LDCC task, and 384 for the other two QA tasks.

There's a big drop for the LDCC accuracy when the max sequence length is shorter, however, the performance for QA tasks is barely influenced by it. It does take more time and computing resources to fine-tune and evaluate when `max_seq_length` is longer. Therefore, it might not be a bad idea to shorten the `max_seq_length` for QA tasks when there's no sufficient resources.

| MODEL | LDCC ACC | LDCC F1 | RCQA EM | RCQA F1 | DDQA EM | DDQA F1 |
|-|-|-|-|-|-|-|
| Laboro-large-unigram, len=128/384 | 94.29 | 94.28 | 73.52 | 76.69 | 74.32 | 85.19 |
| Laboro-large-unigram, len=512 | 97.28 | 97.22 | 73.66 | 76.63 | 76.68 | 86.24 | 
| Laboro-large-BPE, len=128/384 | 92.53 | 92.54 | 73.50 | 75.22 | 87.40 | 92.35 |
| Laboro-large-BPE, len=512 | 97.01 | 96.98 | 73.54 | 75.25 | 86.72 | 92.03 |
| Laboro-base-unigram, len=128/384 | 94.29 | 94.25 | 72.96 | 76.19 | 72.66 | 84.35 |
| Laboro-base-unigram, len=512 | 96.74 | 96.67 | 72.86 | 76.00 | 72.85 | 84.34 |
| Laboro-base-BPE, len=128/384 | 92.93 | 92.89 | 73.41 | 74.70 | 86.72 | 91.14 |
| Laboro-base-BPE, len=512 | 97.55 | 97.60 | 73.26 | 74.79 | 86.42 | 91.19 |
| NICT-BPE, len=128/384 | 92.66 | 92.47 | 77.81 | 79.17 | 88.96 | 92.68 |
| NICT-BPE, len=512 | 96.74 | 96.66 | 78.44 | 79.76 | 89.36 | 93.28 |
| NICT-noBPE, len=128/384 | 92.53 | 92.45 | 76.48 | 77.87 | 88.09 | 92.65 |
| NICT-noBPE, len=512 | 96.33 | 96.18 | 76.74 | 78.20 | 87.21 | 91.58 |

### Unanswerable Questions Threshold

For QA tasks with unanswerable questions, adjusting null_score_diff_threshold can further improve the performance. Judging if the answer exists is actually a classification task, and it is done by comparing the score of the most possible non-null answer to the score of not having an answer. In mathematical language, when snon-null > snull + τ, it predicts a non-null answer. The τ serves as a threshold, and according to Devlin et al., 2019, the threshold that maximizes the F1 for the dev set should be selected.

The table below compares the performance for RCQA task when applying the threshold to not applying the threshold. Although it's not always the case, applying the threshold usually improves the performance.

| MODEL | RCQA EM, w/o best_f1_thresh | RCQA F1, w/o best_f1_thresh | RCQA EM, w/ best_f1_thresh | RCQA F1, w/ best_f1_thresh |
|-|-|-|-|-|
| Laboro-large-unigram | 73.52 | 73.23 | 76.69 | 76.49 |
| Laboro-large-bpe | 73.50 | 48.88 | 75.22 | 51.03 |
| Laboro-base-unigram | 72.96 | 72.66 | 76.19 | 75.96 |
| Laboro-base-bpe | 73.41 | 73.16 | 74.70 | 74.58 |
| UKyoto-large | 75.67 | 75.53 | 77.25 | 77.21 |
| UKyoto-base | 72.87 | 72.90 | 74.44 | 74.48 |
| NICT-BPE | 77.81 | 77.81 | 79.17 | 79.17 |
| NICT-noBPE | 76.48 | 76.59 | 77.87 | 78.26 |
| UTohoku | 75.43 | 75.75 | 75.63 | 76.02 |
| bert-wiki-ja | 73.62 | 73.58 | 76.58 | 76.56 |