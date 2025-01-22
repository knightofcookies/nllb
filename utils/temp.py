import pandas as pd
from datetime import datetime

start = datetime.now()

df = pd.read_csv("../datasets/samanantar_corpus.tsv", engine="python", sep="\t\t\t\t\t")
base = 2500000
max_offset = 50000

with open(
    "../datasets/samanantar_en_filtered.txt", "w", encoding="utf-8"
) as en_file, open(
    "../datasets/samanantar_kha_filtered.txt", "w", encoding="utf-8"
) as kha_file:
    for index in range(max_offset):
        kha_sentence = df.iloc[base + index].kha
        en_sentence = df.iloc[base + index].en

        en_file.write(en_sentence.strip() + "\n")
        kha_file.write(kha_sentence.strip() + "\n")

end = datetime.now()

print("Operation complete. Time taken: ", end - start)
