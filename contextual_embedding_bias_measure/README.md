# Contextual Embedding Bias Measure (adapted version)

This project is based on the [keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure) methodology for evaluating **bias in contextual embeddings** of language models.

The original library was designed for Python 3.7 and older versions of `transformers`, and therefore was not compatible with newer architectures (e.g., ModernBERT, RoBERTa).  
This adapted version introduces modifications that allow the evaluation to run in a modern environment (Python 3.10+, latest `transformers` and `torch`).

## Core Idea

The methodology measures model **bias** by comparing how the model evaluates different candidates (e.g., traits, occupations, or technical skills) within various contexts.  
Instead of using the original plug-in structure, this version separates the process into two clear stages:

1. **Model Execution (`run_bias_extract.py`)**  
   - Loads selected Hugging Face models (e.g., ModernBERT RC1–RC3).  
   - Generates CSV files (`bias_inputs_*.csv`, `raw_pronoun_*.csv`) containing log-probabilities, top-k token predictions, and contextual data for each model.  
   - The results are saved to the `out/` directory.

2. **Results Analysis (`Exposing_Bias_BERT.ipynb`)**  
   - Reads the generated CSV files from `out/`.  
   - Performs aggregation, visualization, and bias metric calculations (e.g., log-probability differences between “He” and “She”).

## Running the Scripts

Example (ATTR mode):

```bash
python notebooks/run_bias_extract.py   --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1   --candidates notebooks/data_lt/positive_traits.txt   --out notebooks/out/bias_inputs_rc1_positive.csv   --device auto
```

Example (PRONOUN mode):

```bash
python notebooks/run_bias_extract.py   --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3   --candidates notebooks/data_lt/in_demand_tech_skills.txt   --out notebooks/out/raw_pronoun_rc3_skills.csv   --mode pronoun   --device auto
```

## Modifications

- Removed dependencies on outdated libraries (`allennlp`, `pytorch-pretrained-bert`).  
- Replaced with `transformers.AutoModelForMaskedLM` and `AutoTokenizer` using `trust_remote_code`.  
- Added a CLI-based runner to support any Hugging Face model.  
- Outputs are saved in `.csv` format for flexible analysis in Jupyter notebooks.

## Source

Methodology based on:  
> Keita Kurita et al., *"Measuring Bias in Contextualized Word Representations"*, 2019.  
> [https://github.com/keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure)