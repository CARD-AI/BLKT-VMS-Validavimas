import os
import json
import hunspell
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

# Initialize Hunspell
hobj = hunspell.HunSpell("data/lt_dictionary/lt_LT.dic", "data/lt_dictionary/lt_LT.aff")


@lru_cache(maxsize=None)
def is_misspell(word):
    if not hobj.spell(word):
        if hobj.suggest(word):
            return True
    return False


def tokenize(text):
    return re.findall(r"\b[\wĄČĘĖĮŠŲŪŽąčęėįšųūž]+\b", text)


def classify_words(text):
    words = tokenize(text)
    valid_words = [w for w in words if len(w) > 2 and w.isalpha()]
    total_words = len(valid_words)

    misspelled = []
    # foreign = []

    for word in tqdm(valid_words):
        if is_misspell(word):
            misspelled.append(word)

    return {
        "misspelled_words": misspelled,
        # "foreign_words": foreign,
        "misspelled_ratio": len(misspelled) / total_words if total_words > 0 else 0.0,
        # "foreign_ratio": len(foreign) / total_words if total_words > 0 else 0.0
    }


def report(results):
    os.makedirs("output", exist_ok=True)
    result_df = pd.DataFrame(results)

    result_df["misspelled_ratio"] = result_df["misspelled_ratio"] * 100
    result_df.iloc[:-2, :].to_csv(
        "output/misspelled_proportion_of_words.csv", index=False
    )
    print(result_df.iloc[:-2, :])

    test_avg = result_df.iloc[:-2, :][
        result_df.iloc[:-2, :]["dataset_name"].str.startswith("lt_test")
    ]["misspelled_ratio"].mean()
    train_avg = result_df.iloc[:-2, :][
        result_df.iloc[:-2, :]["dataset_name"].str.startswith("lt_train")
    ]["misspelled_ratio"].mean()

    print(f"Average non-Lithuanian word ratio (test): {round(test_avg, 2)}%")
    print(f"Average non-Lithuanian word ratio (train): {round(train_avg, 2)}%")


def main():
    results = []
    files = list(Path("data/").rglob("*.jsonl"))
    for file in tqdm(files):
        try:
            texts = []
            tqdm.write(f"Processing {file} ...")
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            texts.append(obj["text"])
                    except json.JSONDecodeError:
                        print(f"Skipping bad JSON in {file}")

            if not texts:
                continue

            combined_text = " ".join(texts)
            stats = classify_words(combined_text)

            results.append(
                {
                    "dataset_name": str(file.relative_to("data/")),
                    "misspelled_ratio": round(stats["misspelled_ratio"], 4),
                    # "foreign_ratio": round(stats["foreign_ratio"], 4),
                    "misspelled_words": sorted(set(stats["misspelled_words"])),
                    # "foreign_words": sorted(set(stats["foreign_words"]))
                }
            )

            tqdm.write(f"Processed {file.name}")

        except Exception as e:
            tqdm.write(f"Error processing {file}: {e}")

    report(results)


if __name__ == "__main__":
    main()
