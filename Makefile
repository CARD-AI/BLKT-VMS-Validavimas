VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip


all: prepare_data run

run: non_lt misspelled finetune

prepare_python:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

prepare_data: makedirs getdata getdict
	$(PYTHON) src/utils/prepare_jsonl.py

non_lt:
	$(PYTHON) src/non_lt.py

misspelled:
	$(PYTHON) src/misspelled.py

finetune:
	$(PYTHON) src/finetune.py

clean:
	rm -rf output
	rm -rf data

makedirs:
	mkdir -p data output

getdata:
	git clone git@github.com:tilde-nlp/MultiLeg-dataset.git
	cp -r MultiLeg-dataset/data/lt/test data/lt_test
	cp -r MultiLeg-dataset/data/lt/train data/lt_train
	rm -rf MultiLeg-dataset

getdict:
	wget --no-verbose -O 'data/1-lt_LT.zip' https://clarin.vdu.lt/xmlui/bitstream/handle/20.500.11821/64/1-lt_LT.zip
	unzip -o 'data/1-lt_LT.zip' -d data/lt_dictionary 
	rm -rf 'data/1-lt_LT.zip'