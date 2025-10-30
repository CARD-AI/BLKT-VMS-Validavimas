VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip


all: prepare_data finetune_modernbertRC1 bias_extract_RC1

prepare_python:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

prepare_data: makedirs getdata
	$(PYTHON) src/utils/prepare_jsonl.py

clean:
	rm -rf data
	rm -rf output_modernbert_rc1

makedirs:
	mkdir -p data

getdata:
	git clone git@github.com:tilde-nlp/MultiLeg-dataset.git
	cp -r MultiLeg-dataset/data/lt/test data/lt_test
	cp -r MultiLeg-dataset/data/lt/train data/lt_train
	rm -rf MultiLeg-dataset

finetune_modernbertRC1:
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/finetune.py --config config/modernbert-RC1.yml

finetune_modernbertRC2:
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/finetune.py --config config/modernbert-RC2.yml
	
finetune_modernbertRC3:
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/finetune.py --config config/roberta-RC3.yml

bias_extract_RC1: 
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc1_positive.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/negative_traits.txt --out bias_data/out/bias_inputs_rc1_negative.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/bias_inputs_rc1_skills.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc1_positive.csv --mode pronoun --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/negative_traits.txt --out bias_data/out/bias_inputs_rc1_negative.csv --mode pronoun --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/bias_inputs_rc1_skills.csv --mode pronoun --device auto

bias_extract_RC2: 
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc2_positive.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2 --candidates bias_data/data_lt/negative_traits.txt --out bias_data/out/bias_inputs_rc2_negative.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/bias_inputs_rc2_skills.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc2_positive.csv --mode pronoun --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2 --candidates bias_data/data_lt/negative_traits.txt --out bias_data/out/bias_inputs_rc2_negative.csv --mode pronoun --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/bias_inputs_rc2_skills.csv --mode pronoun --device auto

bias_extract_RC3: 
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc3_positive.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3 --candidates bias_data/data_lt/negative_traits.txt --out bias_data/out/bias_inputs_rc3_negative.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/bias_inputs_rc3_skills.csv --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc3_positive.csv --mode pronoun --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3 --candidates bias_data/data_lt/negative_traits.txt --out bias_data/out/bias_inputs_rc3_negative.csv --mode pronoun --device auto
	TORCHDYNAMO_DISABLE=1 $(PYTHON) src/bias_extract.py --model neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3 --candidates bias_data/data_lt/in_demand_tech_skills.txt --out bias_data/out/bias_inputs_rc3_skills.csv --mode pronoun --device auto