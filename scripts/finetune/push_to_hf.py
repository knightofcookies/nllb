from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("saved_model/fine_tuned_nllb")
model = AutoModelForSeq2SeqLM.from_pretrained("saved_model/fine_tuned_nllb")

tokenizer.push_to_hub("ahlad/nllb-600M-finetune-en-kha")
model.push_to_hub("ahlad/nllb-600M-finetune-en-kha")
