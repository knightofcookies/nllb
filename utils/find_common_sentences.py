import pandas as pd


def find_and_remove_common_sentences(
    df1, df2, df3, sentence_column_1, sentence_column_2, sentence_column_3
):
    if (
        sentence_column_1 not in df1.columns
        or sentence_column_2 not in df2.columns
        or sentence_column_3 not in df3.columns
    ):
        raise ValueError("One or more sentence columns not found in the dataframes.")

    sentences1 = set(df1[sentence_column_1].dropna())
    sentences2 = set(df2[sentence_column_2].dropna())
    sentences3 = set(df3[sentence_column_3].dropna())

    common_1_2 = list(sentences1.intersection(sentences2))
    common_1_3 = list(sentences1.intersection(sentences3))
    common_2_3 = list(sentences2.intersection(sentences3))
    common_1_2_3 = list(sentences1.intersection(sentences2).intersection(sentences3))

    # df1_filtered = df1[~df1[sentence_column_1].isin(common_1_3)].copy()
    # df2_filtered = df2[~df2[sentence_column_2].isin(common_2_3)].copy()


    return {
        "1_2": common_1_2,
        "1_3": common_1_3,
        "2_3": common_2_3,
        "1_2_3": common_1_2_3,
        # "df1_filtered":df1_filtered,
        # "df2_filtered":df2_filtered
    }


if __name__ == "__main__":
    df1 = pd.read_csv(
        "../datasets/samanantar_4950k_filtered.tsv", engine="python", sep="\t\t\t\t\t"
    )
    df2 = pd.read_csv(
        "../datasets/samanantar_50k_filtered.tsv", engine="python", sep="\t\t\t\t\t"
    )
    df3 = pd.read_csv("../datasets/manual_corpus.csv")

    results = find_and_remove_common_sentences(
        df1, df2, df3, "kha", "kha", "kha"
    )

    common_sentences = results["1_2"],results["1_3"],results["2_3"],results["1_2_3"]

    # df1_filtered = results["df1_filtered"]
    # df2_filtered = results["df2_filtered"]

    with open("output.out", "w", encoding="utf-8") as fp:
        print("For kha:",file=fp)
        print("Common sentences between df1 and df2:", common_sentences[0], file=fp)
        print("Common sentences between df1 and df3:", common_sentences[1], file=fp)
        print("Common sentences between df2 and df3:", common_sentences[2], file=fp)
        print("Common sentences between df1, df2, and df3:", common_sentences[3], file=fp)
        
    # print("df1 filtered shape:", df1_filtered.shape)
    # print("df2 filtered shape:", df2_filtered.shape)


    # df1_filtered.to_csv("samanantar_4950k.csv", index=False, sep="\x1f")
    # df2_filtered.to_csv("samanantar_50k.csv", index=False, sep="\x1f")
