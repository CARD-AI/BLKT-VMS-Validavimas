# MLKV VMS

MLKVM modelio kokybės validavimas pagal vieną iš GLUE (angl. General Language Understanding Evaluation) vertinimo metodikoje (https://gluebenchmark.com/) numatytų vertinimo užduočių: įvardytų esybių atpažinimas.

## Modelis

Validuoti modeliai:
- ModernBERT Stage 3 RC 1: [neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1](https://huggingface.co/neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1)
- ModernBERT Stage 3 RC 2: [neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2](https://huggingface.co/neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2)
- RoBERTa Stage 3 RC 3: [neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3](https://huggingface.co/neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3)

## Reikalavimai

- python3
- make
- Huggingface sesija (prisijungiant su huggingface-cli komanda)

8GB VRAM memory

## Duomenų paruošimas

Norint pradėti dirbti su duomenimis, mum reikia prieigos prie:
- Lithuanian jsonl Named Entity Recognition datasets for train and test sets (https://github.com/tilde-nlp/MultiLeg-dataset/tree/main, folder data/lt/);

Download the files from corresponding links above and place them in data/folder. In total you should have 3 folders:
- data/lt_dictionary
- data/lt_test
- data/lt_train

To automatically download data, run:

```
make getdata
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

Finetune base MLKV model and evaluate it using our dataset
```
make finetune_modernbertRC1
```

The resulting evaluation metrics are saved in the `output_modernbert_rc1` folder.


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

## Contextual Embedding Bias Measure (adapted version)

This project is based on the [keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure) methodology for evaluating **bias in contextual embeddings** of language models.

The original library was designed for Python 3.7 and older versions of `transformers`, and therefore was not compatible with newer architectures (e.g., ModernBERT, RoBERTa).  
This adapted version introduces modifications that allow the evaluation to run in a modern environment (Python 3.10+, latest `transformers` and `torch`).

### Core Idea

The methodology measures model **bias** by comparing how the model evaluates different candidates (e.g., traits, occupations, or technical skills) within various contexts.  
Instead of using the original plug-in structure, this version separates the process into two clear stages:

1. **Model Execution (`bias_extract.py`)**  
   - Loads selected Hugging Face models (e.g., ModernBERT RC1-RC3).  
   - Generates CSV files (`bias_inputs_*.csv`, `raw_pronoun_*.csv`) containing log-probabilities, top-k token predictions, and contextual data for each model.  
   - The results are saved to the `out/` directory.

2. **Results Analysis (`bias_processing.ipynb`)**. 
   - Reads the generated CSV files from `out/`.  
   - Performs aggregation, visualization, and bias metric calculations (e.g., log-probability differences between “He” and “She”).

### Running the Scripts

To run all bias scripts, use the command
```bash
make bias_extract_all
```

To run bias extraction scripts without make, direct python calls can be used.


Example (ATTR mode):

```bash
python bias_data/run_bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc1_positive.csv --device auto
```

Example (PRONOUN mode):

```bash
python bias_data/run_bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/raw_pronoun_rc1_skills.csv --mode pronoun --device auto
```

### Modifications

- Removed dependencies on outdated libraries (`allennlp`, `pytorch-pretrained-bert`).  
- Replaced with `transformers.AutoModelForMaskedLM` and `AutoTokenizer` using `trust_remote_code`.  
- Added a CLI-based runner to support any Hugging Face model.  
- Outputs are saved in `.csv` format for flexible analysis in Jupyter notebooks.

### Source

Methodology based on:  
> Keita Kurita et al., *"Measuring Bias in Contextualized Word Representations"*, 2019.  
> [https://github.com/keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure)