# MLKVM validavimo sprendimas

MLKVM kokybės validavimas pagal vieną iš GLUE (angl. General Language Understanding Evaluation) vertinimo metodikoje (https://gluebenchmark.com/) numatytų vertinimo užduočių: įvardytų esybių atpažinimas.

## Modelis

Validuoti modeliai:
- ModernBERT Stage 3 RC 1: [neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1](https://huggingface.co/neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1)
- ModernBERT Stage 3 RC 2: [neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2](https://huggingface.co/neurotechnology/BLKT-ModernBert-MLM-Stage3-RC2)
- RoBERTa Stage 3 RC 3: [neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3](https://huggingface.co/neurotechnology/BLKT-RoBerta-MLM-Stage3-RC3)

## Reikalavimai

- python3.12
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

Kiekvienas modelis apmokymo metu yra įvertinamas šiomis metrikomis:

1. **Teksto vienetų (angl. *tokens*) lygiu:**
   - Tikslumas  
   - Preciziškumas  
   - Atkūrimo statistika  
   - F1 statistika  

2. **Esybių (angl. *entities*) lygiu:**  
   Modelio veikimas vertinamas pagal preciziškumą, atkūrimo statistiką ir F1 rodiklį, taikant skirtingus atitikimo kriterijus:
   - **Tikslus atitikimas** (*exact match*) – esybės tipas ir visa jos sritis (pradžia ir pabaiga) turi tiksliai sutapti.  
   - **Persidengiantis atitikimas** (*overlap match*) – laikoma teisinga, jei bent vienas žodis sutampa su ta pačia žyme.  
   - **Sąjungos atitikimas** (*union match*) – laikoma teisinga, jei prognozė bet kaip persidengia su tikrąja esybe.  



## MLKVM šališkumo matavimas

Šis projektas paremtas [keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure) metodologija, skirta vertinti **šališkumą kontekstiniuose kalbos modelių įterpiniuose**.

Originali biblioteka buvo sukurta Python 3.7 versijai ir senesnėms `transformers` bibliotekos versijoms, todėl nebuvo suderinama su naujesnėmis architektūromis (pvz., ModernBERT, RoBERTa).  
Ši adaptuota versija pateikia pakeitimus, leidžiančius vykdyti vertinimą šiuolaikinėje aplinkoje (Python 3.10+, naujausios `transformers` ir `torch` versijos).

### Pagrindinė Idėja

Metodologija matuoja modelio **šališkumą**, lygindama, kaip modelis vertina skirtingus kandidatus (pvz., savybes, profesijas ar techninius įgūdžius) įvairiuose kontekstuose.  
Vietoje originalios papildinio struktūros, ši versija procesą padalija į du aiškius etapus:

1. **Modelio Vykdymas (`bias_extract.py`)**  
   - Įkelia pasirinktus Hugging Face modelius (pvz., ModernBERT RC1–RC3).  
   - Generuoja CSV failus (`bias_inputs_*.csv`, `raw_pronoun_*.csv`), kuriuose pateikiamos logaritminės tikimybės, top-k žetonų prognozės ir kontekstiniai duomenys kiekvienam modeliui.  
   - Rezultatai išsaugomi kataloge `out/`.

2. **Rezultatų Analizė (`bias_processing.ipynb`)**  
   - Nuskaito sugeneruotus CSV failus iš `out/` katalogo.  
   - Atlieka duomenų agregavimą, vizualizaciją ir šališkumo metrikų skaičiavimus (pvz., logaritminių tikimybių skirtumai tarp „He“ ir „She“).

### Skriptų Vykdymas

Norint paleisti visus šališkumo skriptus, naudokite komandą:
```bash
make bias_extract_all
```

Norint vykdyti šališkumo ištraukimo skriptus be make, galima naudoti tiesioginius python iškvietimus.

Pavyzdys (ATTR režimas):
```bash
python bias_data/run_bias_extract.py --model neurotechnology/BLKT-ModernBert-MLM-Stage3-RC1 --candidates bias_data/data_lt/positive_traits.txt --out bias_data/out/bias_inputs_rc1_positive.csv --device auto
```

Pavyzdys (PRONOUN režimas):
```bash
python bias_data/run_bias_extract.py --model neurotechnology/BLKT-ModernBert_
```

### Pakeitimai

- Pašalintos priklausomybės nuo pasenusių bibliotekų (`allennlp`, `pytorch-pretrained-bert`).  
- Pakeista į `transformers.AutoModelForMaskedLM` ir `AutoTokenizer`, naudojant `trust_remote_code`.  
- Pridėtas CLI pagrindu veikiantis paleidiklis, palaikantis bet kurį Hugging Face modelį.  
- Išvestys išsaugomos `.csv` formatu, kad būtų patogu analizuoti Jupyter užrašuose.

### Šaltinis

Metodologijos pagrindas:  
> Keita Kurita ir kt., *„Measuring Bias in Contextualized Word Representations“*, 2019.  
> [https://github.com/keitakurita/contextual_embedding_bias_measure](https://github.com/keitakurita/contextual_embedding_bias_measure)
