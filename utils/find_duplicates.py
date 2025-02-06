import pandas as pd
import sys


def find_duplicate_sentences(df, sentence_column):
    if sentence_column not in df.columns:
        raise ValueError("Specified sentence column not found in the dataframe.")

    sentences = df[sentence_column].dropna().tolist()
    sentence_counts = {}
    duplicates = {}

    for i, sentence in enumerate(sentences):
        if sentence in sentence_counts:
            sentence_counts[sentence].append(i)  # Append index of duplicate occurrence
            if sentence not in duplicates:
                duplicates[sentence] = sentence_counts[sentence]
        else:
            sentence_counts[sentence] = [i]  # Store first index of sentence

    return duplicates


def find_duplicate_sentences_with_context(df, sentence_column):
    if sentence_column not in df.columns:
        raise ValueError("Specified sentence column not found in the dataframe.")

    duplicates = find_duplicate_sentences(df, sentence_column)

    duplicates_with_context = {}
    for sentence, indices in duplicates.items():
        duplicates_with_context[sentence] = df.iloc[indices].to_dict("records")

    return duplicates_with_context


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep="\t\t\t\t\t", engine="python")

    duplicates = find_duplicate_sentences(df, "en")
    if duplicates:
        print(f"Duplicates found : {len(duplicates)}")
    else:
        print("No duplicates found.")
