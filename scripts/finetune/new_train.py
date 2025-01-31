import re
import pandas as pd
import random
import sys
import os
import unicodedata
from sacremoses import MosesPunctNormalizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset, load_from_disk


TOKENIZED_DATASET_PATH = "./tokenized_samanantar_4950k"


model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.cuda()


if os.path.exists(TOKENIZED_DATASET_PATH):
    print("Loading tokenized dataset from disk...")
    tokenized_dataset = load_from_disk(TOKENIZED_DATASET_PATH)

else:
    print("Tokenizing dataset...")

    df_train = pd.read_csv(
        "../../datasets/samanantar_4950k_filtered.tsv",
        sep="\t\t\t\t\t",
        engine="python",
    )
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]

    def get_non_printing_char_replacer(replace_by: str = " "):
        non_printable_map = {
            ord(c): replace_by
            for c in (chr(i) for i in range(sys.maxunicode + 1))
            if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
        }

        def replace_non_printing_char(line) -> str:
            return line.translate(non_printable_map)

        return replace_non_printing_char

    replace_nonprint = get_non_printing_char_replacer(" ")

    def preproc(text):
        clean = mpn.normalize(text)
        clean = replace_nonprint(clean)
        clean = unicodedata.normalize("NFKC", clean)
        return clean

    LANGS = [("en", "eng_Latn"), ("kha", "vie_Latn")]

    def preprocess_dataset(dataset):
        dataset["kha"] = [preproc(text) for text in dataset["kha"]]
        dataset["en"] = [preproc(text) for text in dataset["en"]]
        return dataset

    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.map(preprocess_dataset, batched=True)

    def tokenize_dataset(batch):
        (l1, long1), (l2, long2) = random.sample(LANGS, 2)
        tokenizer.src_lang = long1
        inputs = tokenizer(batch[l1], padding=True, truncation=True, max_length=128)
        tokenizer.src_lang = long2
        labels = tokenizer(batch[l2], padding=True, truncation=True, max_length=128)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize_dataset, batched=True)

    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)
    print("Tokenized dataset saved to disk.")


training_args = Seq2SeqTrainingArguments(
    output_dir="./saved_model/fine_tuned_nllb",
    per_device_train_batch_size=32,
    num_train_epochs=1,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="no",
    prediction_loss_only=True,
    fp16=True,
    warmup_steps=1000,
    learning_rate=1e-4,
    weight_decay=1e-3,
    adafactor=True,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./saved_model/fine_tuned_nllb")
tokenizer.save_pretrained("./saved_model/fine_tuned_nllb")
