# import gdown

# url = "https://drive.google.com/file/d/1kT4xR8ywdtu887PeH_3suLQJEoEx7PTl/view?usp=sharing"
# output = "news_corpus.tsv"
# gdown.download(url=url, output=output, fuzzy=True)

from datasets import load_dataset

dataset = load_dataset("lilferrit/wmt14-short")

with open("english_wmt14.tsv", "w", encoding="utf-8") as f:
    f.write("sentence\n")
    for item in dataset["train"]["translation"]:
        f.write(f"{item['en']}\n")
