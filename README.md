# Exploring Span Representations in Neural Coreference Resolution

This repository contains code for our paper. We attempt to probe to what extent can span representations encode coreference relations. We also question whether if span representations are able to encode long-range coreference phenomena effectively, or are they just simply modelling local coreference relations. We extend the implementation of [BERT for Coreference Resolution](https://github.com/mandarjoshi90/coref). The source code is located under `src`.

## Setup
* Install python3 requirements: `pip install -r requirements.txt`
* Export path to OntoNotes directory: `export data_dir=</path/to/data_dir>`
* `/setup_all.sh`: This script builds the custom kernels.
* `setup_training.sh`: This script preprocesses the OntoNotes corpus and download the original BERT models.

## Pre-trained Models
The pretrained models can be downloaded using `download_pretrained.sh <model_name>` (i.e. `bert_base` or `bert_large`; this assumes that `$data_dir` is set).

## Extracting Span Representations
To extract the span representations in .h5 format, run `extract_span.py` and `extract_span_baseline.py` for the baseline. Here is a sample code:
```
python3 extract_span.py bert_base $data_dir/<input.jsonlines> $data_dir span_representation_bert_base
```

## Running Probing Experiments
The extracted .h5 files can be used to run probing experiments using `train_baseline.py` and `train_probe.py`. Here is a sample code:
```
python3 train_baseline.py --train_data </path/to/train_data> --val_data </path/to/val_data> --test_data </path/to/test_data> --exp_name <test_experiment_name> --cnn_context 1 --embed_dim 1024
```