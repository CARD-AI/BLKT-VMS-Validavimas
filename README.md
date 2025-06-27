# BLKT VMS

This project validates BLKT dataset.

## Model

We used RoBERTa base model: `neurotechnology/BLKT-RoBERTa-MLM-Stage2-Intermediate`

## Prerequisites

- python3
- make
- Huggingface session (logged in through huggingface-cli command)

## Data preparation

To start working with the data, we need to access:
- Lithuanian dictionaries to calculate proportions (https://clarin.vdu.lt/xmlui/handle/20.500.11821/64, archive 1-lt_LT.zip);
- Lithuanian jsonl Named Entity Recognition datasets for train and test sets (https://github.com/tilde-nlp/MultiLeg-dataset/tree/main, folder data/lt/);

Download the files from corresponding links above and place them in data/folder. In total you should have 3 folders:
- data/lt_dictionary
- data/lt_test
- data/lt_train

To automatically download data, run:

```
make getdata
make getdict
```

### Convert the data (jsonl to conll)

Conll is the usual format for using the fine-tuning of Named Entity Recognition (NER) tasks. For this, we are using the conversion scripts in `src/utils/jsonl_converter.py`

To convert the data, run: 

```
python src/utils/prepare_jsonl.py
```

To download and convert data in a single step, run:
```
make prepare_data
```
## Run

Create a virtual environment and install dependencies:
```
make prepare_python
```

Download datasets, prepare them, run fine tuning and evaluation of datasets lithuanian metrics, run:
```
make all
```

### Finetune the model

Run model fine tuning
```
make finetune
```

The resulting evaluation metrics are saved in the output/ folder.

### Evaluate lithuanian datasets

To evaluate how many lithuanian words are misspelled in the dataset, run:
```
make misspelled
```

To evaluate how many non lithuanian words are in the dataset, run:
```
make non_lt
```

## Evaluation Metrics

_Token-Level:_
- Accuracy
- Precision
- Recall
- F1-score

_Entity-Level:_
- Exact Match: Full span and type must match
- Overlap Match: Any token overlap with same label counts
- Union Match: Prediction overlaps in any way with true entity

_Each of these includes:_
- Precision
- Recall
- F1-score

