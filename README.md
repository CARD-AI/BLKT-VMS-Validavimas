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

### Skaičiavimo resursų reikalavimai

Atlikti skaičiavimams reikia 8GB vaizdo plokštės atminties.

## Duomenų paruošimas

Norint pradėti dirbti su duomenimis, mum reikia prieigos prie:
- Lithuanian jsonl Named Entity Recognition duomenų rinkinio su mokinimo ir testavimo duomenimis (https://github.com/tilde-nlp/MultiLeg-dataset/tree/main, direktorijos data/lt/);

Duomenys turi būti atsiųsti į direktorijas:
- data/lt_test
- data/lt_train

Automatiškai galima parsisiųsti duomenis naudojant komandą:
```bash
make getdata
```

### Duomenų pavertimas iš `jsonl` formato į `conll` formatą

Modelį apmokinant įvardytų esybių atpažinimo uždaviniui naudojamas Conll formatas. Kadangi mūsų duomenys yra `jsonl` formato, juos pasiverčiame į `conll` formatą naudodami `src/utils/jsonl_converter.py` kodą.

Norint atlikti pavertimą, paleidžiame kodą:

```bash
python src/utils/prepare_jsonl.py
```

Norint atsisiųsti `jsonl` formato duomenis ir automatiškai juos pasiversti į `conll` formatą, naudojame komandą:
```bash
make prepare_data
```

## Programos paleidimas

Sukuriame virtualią python aplinką ir įdiegiame naudojamas bibliotekas su komanda:
```bash
make prepare_python
```

Parsisiunčiame duomenis, juos paruošiame, paleidžiame modelio apmokinimo kodą ir atliekame įvertinimą su komanda:
```bash
make all
```

### Modelio apmokinimas

Modelį apmokiname įvardytų esybių atpažinimo uždaviniui ir atliekame įvertinimą su komanda:
```bash
make finetune_modernbertRC1
```

## Skaičiuojami įvertinimo rodikliai

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