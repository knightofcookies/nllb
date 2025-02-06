import sys
import pandas as pd


def filter_sentence_length(input_file, output_file):
    df = pd.read_csv(input_file)

    filtered_df = df[
        (df["en"].str.split().str.len().between(5, 30))
        & (df["kha"].str.split().str.len().between(5, 30))
    ]

    filtered_df.to_csv(output_file, index=False)

    print(f"Original number of sentences: {len(df)}")
    print(f"Filtered number of sentences: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} sentence pairs")


filter_sentence_length(sys.argv[1], sys.argv[2])
