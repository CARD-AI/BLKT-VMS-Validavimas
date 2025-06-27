import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_conll(filepath):
    tokens, labels, examples = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": labels})
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) != 2:
                    print(f"Skipping malformed line {line_num} in {filepath}: '{line}'")
                    continue
                token, tag = parts
                tokens.append(token)
                labels.append(tag)
        if tokens:
            examples.append({"tokens": tokens, "ner_tags": labels})
    return examples


def load_conll_folder(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".conll"):
            full_path = os.path.join(folder_path, filename)
            examples = read_conll(full_path)
            all_data.extend(examples)
    return all_data


def load_data(train_dir, test_dir):
    train_data = load_conll_folder(train_dir)
    test_data = load_conll_folder(test_dir)

    # Extract label list
    unique_labels = sorted({label for ex in train_data for label in ex["ner_tags"]})
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Convert string labels to IDs
    for dataset in (train_data, test_data):
        for ex in dataset:
            ex["labels"] = [label2id[tag] for tag in ex["ner_tags"]]

    return {
        "dataset": dataset,
        "label2id": label2id,
        "id2label": id2label,
        "train_data": train_data,
        "test_data": test_data,
    }


def tokenize_and_align(example, tokenizer):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized.word_ids()

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized


def align_predictions(predictions, label_ids, id2label):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_preds, out_labels = [], []

    for i in range(batch_size):
        pred_labels = []
        true_labels = []
        for j in range(seq_len):
            if label_ids[i][j] != -100:
                pred_labels.append(id2label[preds[i][j]])
                true_labels.append(id2label[label_ids[i][j]])
        out_preds.append(pred_labels)
        out_labels.append(true_labels)

    return out_preds, out_labels


def extract_entities(label_seq):
    entities = []
    start, end, label = None, None, None
    for i, tag in enumerate(label_seq):
        if tag.startswith("B-"):
            if label is not None:
                entities.append((start, end, label))
            start = i
            end = i + 1
            label = tag[2:]
        elif tag.startswith("I-") and label == tag[2:]:
            end = i + 1
        else:
            if label is not None:
                entities.append((start, end, label))
                start, end, label = None, None, None
    if label is not None:
        entities.append((start, end, label))
    return entities


def compute_span_matches(pred_spans, true_spans, mode="exact"):
    tp, fp, fn = 0, 0, 0
    used = set()

    for ps in pred_spans:
        matched = False
        for i, ts in enumerate(true_spans):
            if ts[2] != ps[2]:  # labels must match
                continue

            if mode == "exact" and ps == ts:
                matched = True
                used.add(i)
                break

            elif mode == "overlap":
                if max(ps[0], ts[0]) < min(ps[1], ts[1]):
                    matched = True
                    used.add(i)
                    break

            elif mode == "union":
                if ps[0] <= ts[1] and ps[1] >= ts[0]:
                    matched = True
                    used.add(i)
                    break

        if matched:
            tp += 1
        else:
            fp += 1

    fn = len([ts for i, ts in enumerate(true_spans) if i not in used])
    return tp, fp, fn


def compute_metrics(p, id2label):
    predictions, labels = p
    preds, refs = align_predictions(predictions, labels, id2label)

    # Matthews Correlation Coefficient
    flat_preds = [p for seq in preds for p in seq]
    flat_refs = [r for seq in refs for r in seq]

    mcc = matthews_corrcoef(flat_refs, flat_preds)
    # print(f'Matthews Correlation Coefficient: {mcc}')

    # Confusion Matrix
    cm = confusion_matrix(flat_refs, flat_preds)
    labels = np.unique(flat_refs)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # print("Confusion Matrix (per class):")

    # token-level scores
    token_accuracy = accuracy_score(refs, preds)
    token_precision = precision_score(refs, preds)
    token_recall = recall_score(refs, preds)
    token_f1 = f1_score(refs, preds)

    # span-level scores
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    overlap_tp, overlap_fp, overlap_fn = 0, 0, 0
    union_tp, union_fp, union_fn = 0, 0, 0

    for pred_seq, ref_seq in zip(preds, refs):
        pred_spans = extract_entities(pred_seq)
        ref_spans = extract_entities(ref_seq)

        tp, fp, fn = compute_span_matches(pred_spans, ref_spans, mode="exact")
        exact_tp += tp
        exact_fp += fp
        exact_fn += fn

        tp, fp, fn = compute_span_matches(pred_spans, ref_spans, mode="overlap")
        overlap_tp += tp
        overlap_fp += fp
        overlap_fn += fn

        tp, fp, fn = compute_span_matches(pred_spans, ref_spans, mode="union")
        union_tp += tp
        union_fp += fp
        union_fn += fn

    def safe_div(a, b):
        return a / b if b else 0.0

    def scores(tp, fp, fn):
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = (
            safe_div(2 * precision * recall, precision + recall)
            if (precision + recall)
            else 0.0
        )
        return precision, recall, f1

    exact_p, exact_r, exact_f1 = scores(exact_tp, exact_fp, exact_fn)
    overlap_p, overlap_r, overlap_f1 = scores(overlap_tp, overlap_fp, overlap_fn)
    union_p, union_r, union_f1 = scores(union_tp, union_fp, union_fn)

    return {
        "token_accuracy": token_accuracy,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
        "entity_exact_precision": exact_p,
        "entity_exact_recall": exact_r,
        "entity_exact_f1": exact_f1,
        "entity_overlap_precision": overlap_p,
        "entity_overlap_recall": overlap_r,
        "entity_overlap_f1": overlap_f1,
        "entity_union_precision": union_p,
        "entity_union_recall": union_r,
        "entity_union_f1": union_f1,
        "matthews_coefficient": mcc,
    }


def eval_metrics(trainer, train_dataset, test_dataset, output_dir):
    # Final Evaluation
    final_predictions, _, metrics = trainer.predict(test_dataset)

    # Evaluate on test dataset
    test_preds, _, test_metrics = trainer.predict(test_dataset)

    # Evaluate on train dataset
    train_preds, _, train_metrics = trainer.predict(train_dataset)

    # Convert to DataFrames
    test_df = pd.DataFrame([test_metrics], index=["Test"])
    train_df = pd.DataFrame([train_metrics], index=["Train"])

    # Combine into a single summary table
    results_df = pd.concat([train_df, test_df])
    # print("\nEvaluation Metrics (Train vs Test):\n")
    # print(results_df.T.round(4))
    results_df.T.to_csv(f"{output_dir}/evaluation_metrics_results.csv", index=True)

    # Reformatting data for grouped bar plot
    metrics = results_df.T[:-3].index.tolist()
    train_values = results_df.T["Train"][:-3].tolist()
    test_values = results_df.T["Test"][:-3].tolist()

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([p - width / 2 for p in x], train_values, width, label="Train")
    ax.bar([p + width / 2 for p in x], test_values, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Train vs Test Metrics (Grouped by Metric)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_metrics_plot.png", dpi=600)


def eval_misclassified(trainer, tokenizer, test_dataset, id2label):
    # Step 1: Get predictions and aligned labels
    predictions, label_ids, _ = trainer.predict(test_dataset)
    preds, refs = align_predictions(predictions, label_ids, id2label)

    # Step 2: Decode tokens for each example
    print("Misclassified NER Samples:\n")
    max_display = 10
    shown = 0

    for i in range(len(test_dataset)):
        # Get input_ids for the current example
        input_ids = test_dataset[i]["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Get aligned predicted and true labels
        pred_labels = preds[i]
        true_labels = refs[i]

        # Filter mismatches (skip padding)
        mismatches = [
            (tok, pred, true)
            for tok, pred, true in zip(tokens, pred_labels, true_labels)
            if pred != true and true != "O" and tok not in tokenizer.all_special_tokens
        ]

        if mismatches:
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Text: {decoded_text}")
            print("Mismatched Tokens:")
            for tok, pred, true in mismatches:
                print(f"  Token: {tok:15} | Predicted: {pred:10} | True: {true}")
            print("-" * 60)
            shown += 1

        if shown >= max_display:
            break


def main():
    model_checkpoint = "neurotechnology/BLKT-RoBERTa-MLM-Stage2-Intermediate"
    output_dir = "output"
    trainer_output_dir = "output/model"
    train_dir = "data/conll_train/"
    test_dir = "data/conll_test/"
    num_epochs = 10

    data = load_data(train_dir, test_dir)
    # Tokenization and Alignment
    # Put the token using !huggingface-cli login
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  # , token=HF_TOKEN)
    train_dataset = Dataset.from_list(data["train_data"]).map(
        lambda x: tokenize_and_align(x, tokenizer), batched=False
    )
    test_dataset = Dataset.from_list(data["test_data"]).map(
        lambda x: tokenize_and_align(x, tokenizer), batched=False
    )

    # Model Setup
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(data["label2id"]),
        id2label=data["id2label"],
        label2id=data["label2id"],
        use_safetensors=True,
    )

    # Training Setup
    args = TrainingArguments(
        output_dir=trainer_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    # train the model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, data["id2label"]),
    )

    trainer.train()

    eval_metrics(trainer, train_dataset, test_dataset, output_dir)
    eval_misclassified(trainer, tokenizer, test_dataset, data["id2label"])


if __name__ == "__main__":
    main()
