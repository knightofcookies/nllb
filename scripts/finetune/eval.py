from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import Dataset
from datetime import datetime
import torch

start = datetime.now()

model_name = "../../saved_model/fine_tuned_nllb"

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

translator_nllb = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="vie_Latn",
    tgt_lang="eng_Latn",
    max_length=128,
    device=device.index if torch.cuda.is_available() else -1,
)

input_file = "../../datasets/samanantar_filtered_kha.txt"
output_file = "../../datasets/translated_samanantar_filtered_kha.txt"

with open(input_file, "r", encoding="utf-8") as infile:
    lines = [line.strip() for line in infile if line.strip()]

dataset = Dataset.from_dict({"source_text": lines})


def translate_batch(batch):
    translations = translator_nllb(batch["source_text"])
    batch["translated_text"] = [t["translation_text"] for t in translations]
    return batch


translated_dataset = dataset.map(translate_batch, batched=True, batch_size=1024)

with open(output_file, "w", encoding="utf-8") as outfile:
    for line in translated_dataset["translated_text"]:
        outfile.write(line + "\n")

print("Translation complete! Check the output file:", output_file)
end = datetime.now()
print("Time taken:", end - start)
