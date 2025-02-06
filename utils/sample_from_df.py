import sys
import pandas as pd


def split_dataframe_randomly(df, num_samples=50000):
    if len(df) < num_samples:
        print(
            f"Warning: DataFrame has {len(df)} rows, which is less than the requested {num_samples} sample size."
        )
        return None, None

    sampled_df = df.sample(n=num_samples, random_state=42)
    remaining_df = df.drop(sampled_df.index)

    return sampled_df, remaining_df


if __name__ == "__main__":
    original_df = pd.read_csv(
        sys.argv[1],
        sep="\t\t\t\t\t",
        engine="python",
    )

    sample_size = 50000
    sampled_df, remaining_df = split_dataframe_randomly(original_df, sample_size)

    sampled_df.to_csv(sys.argv[2], sep="\x1f", index=False)
    remaining_df.to_csv(sys.argv[3], sep="\x1f", index=False)

    if sampled_df is not None and remaining_df is not None:
        print("Sampled DataFrame:")
        print(sampled_df.head())
        print("\nRemaining DataFrame:")
        print(remaining_df.head())
        print("\nShape of Sampled DataFrame: ", sampled_df.shape)
        print("Shape of Remaining DataFrame: ", remaining_df.shape)

        print("\nOriginal DataFrame Size: ", len(original_df))
        print(
            "Combined size of sampled and remaining DataFrame: ",
            len(sampled_df) + len(remaining_df),
        )
