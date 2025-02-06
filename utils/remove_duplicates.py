import sys
import pandas as pd


def remove_duplicate_sentences(df, sentence_column, keep="first"):
    if sentence_column not in df.columns:
        raise ValueError("Specified sentence column not found in the dataframe.")

    df_no_duplicates = df.drop_duplicates(subset=[sentence_column], keep=keep)
    return df_no_duplicates


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep="\t\t\t\t\t", engine="python")

    df_no_dupes_first = remove_duplicate_sentences(df, sys.argv[3], keep="first")
    df_no_dupes_first.to_csv(sys.argv[2], sep="\x1f", index=False)
