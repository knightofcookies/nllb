from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import Dataset
from datetime import datetime
import torch

start = datetime.now()

model_name = "./saved_model/fine_tuned_nllb"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Ensure the pipeline uses the correct device
translator_nllb = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="vie_Latn",
    tgt_lang="eng_Latn",
    max_length=128,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
)

input_file = "../../datasets/samanantar_50k_filtered.tsv"
output_file = "../../datasets/translated_samanantar_50k_filtered_remaining.tsv"

# Read the TSV file with the custom delimiter
with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

# Extract the 'kha' column using the custom delimiter
header = lines[0].strip().split("\t\t\t\t\t")
data = [line.strip().split("\t\t\t\t\t") for line in lines[41386:]]
kha_column = [
    row[header.index("kha")] for row in data if row[header.index("kha")].strip()
]

dataset = Dataset.from_dict({"source_text": kha_column})


def translate_batch(batch):
    translations = translator_nllb(batch["source_text"])
    batch["translated_text"] = [t["translation_text"] for t in translations]
    return batch


# Translate the dataset in batches
translated_dataset = dataset.map(translate_batch, batched=True, batch_size=1024)

# Write the translated data to a new TSV file with the custom delimiter
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    # Write the header
    outfile.write("\t\t\t\t\t".join(header + ["translated_kha"]) + "\n")
    # Write the data
    for i, row in enumerate(data):
        if i < len(translated_dataset["translated_text"]):
            translated_text = translated_dataset["translated_text"][i]
        else:
            translated_text = ""  # In case the lengths don't match
        outfile.write("\t\t\t\t\t".join(row + [translated_text]) + "\n")

print("Translation complete! Check the output file:", output_file)
end = datetime.now()
print("Time taken:", end - start)
