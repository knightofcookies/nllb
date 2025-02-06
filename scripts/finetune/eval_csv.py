from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import Dataset
from datetime import datetime
import torch
import pandas as pd

start = datetime.now()

model_name = "./saved_model/fine_tuned_nllb"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator_nllb = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="vie_Latn",
    tgt_lang="eng_Latn",
    max_length=128,
    device=0 if torch.cuda.is_available() else -1,
)

input_file = "../../datasets/manual_corpus.csv"
output_file = "../../datasets/translated_manual_corpus.csv"

df = pd.read_csv(input_file)

kha_column = df['kha'].dropna().tolist()

dataset = Dataset.from_dict({"source_text": kha_column})

def translate_batch(batch):
    translations = translator_nllb(batch["source_text"])
    batch["translated_text"] = [t["translation_text"] for t in translations]
    return batch

translated_dataset = dataset.map(translate_batch, batched=True, batch_size=1024)

df['translated_kha'] = pd.Series(translated_dataset["translated_text"])

df.to_csv(output_file, index=False, encoding='utf-8')

print("Translation complete! Check the output file:", output_file)
end = datetime.now()
print("Time taken:", end - start)
