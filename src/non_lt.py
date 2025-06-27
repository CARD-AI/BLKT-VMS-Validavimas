import os
import pandas as pd
import hunspell
from tqdm import tqdm
from pathlib import Path
import json
from functools import lru_cache

# Initialize Hunspell with our Lithuanian dictionary
hobj = hunspell.HunSpell("data/lt_dictionary/lt_LT.dic", "data/lt_dictionary/lt_LT.aff")


@lru_cache(maxsize=None)
def is_lt_word(word):
    return hobj.spell(word)


def get_non_lt_info(text):
    words = text.split()
    valid_words = [word for word in words if len(word) > 2 and word.isalpha()]

    non_lt_words = [word for word in valid_words if not is_lt_word(word)]

    if len(valid_words) == 0:
        return 0.0, []

    ratio = len(non_lt_words) / len(valid_words)
    return ratio, non_lt_words


def report(results):
    os.makedirs("output", exist_ok=True)
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("output/non_lt_proportion_of_word.csv", index=False)

    print(summary_df)
    print(summary_df.iloc[0, 2])

    test_avg = summary_df[summary_df["dataset_name"].str.startswith("lt_test")][
        "non_lt_word_ratio"
    ].mean()
    train_avg = summary_df[summary_df["dataset_name"].str.startswith("lt_train")][
        "non_lt_word_ratio"
    ].mean()

    print(f"Average non-Lithuanian word ratio (lt_test): {round(test_avg, 2)}%")
    print(f"Average non-Lithuanian word ratio (lt_train): {round(train_avg, 2)}%")


def main():
    results = []

    # Find all .jsonl files in data/ and subfolders
    files = list(Path("data/").rglob("*.jsonl"))
    for file in tqdm(files):
        try:
            all_texts = []
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "text" in obj:
                            all_texts.append(obj["text"])
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in {file}")

            if not all_texts:
                print(f"Skipping {file}: no 'text' entries found")
                continue

            combined_text = " ".join(all_texts)

            # Apply your analysis on the whole combined text
            non_lt_ratio, non_lt_words = get_non_lt_info(combined_text)

            results.append(
                {
                    "dataset_name": str(file.relative_to("data/")),
                    "non_lt_word_ratio": round(non_lt_ratio, 4) * 100,
                    "non_lt_words": sorted(set(non_lt_words)),
                }
            )

            # print(f"Processed {dataset_name}")

        except Exception as e:
            print(f"Error processing {file}: {e}")
    report(results)


if __name__ == "__main__":
    main()
