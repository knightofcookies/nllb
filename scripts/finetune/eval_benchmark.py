import time
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from tqdm import tqdm

# TODO ONNX doesn't work and takes forever
# TODO Use Dataset with pipeline

test_df = pd.read_csv("../../datasets/processed/smc.csv")

model_name = "./saved_model/fine_tuned_nllb"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="vie_Latn")
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def benchmark_translation_method(method, name):
    if name == "HuggingFace Pipeline":
        translator = pipeline(
            "translation",
            model=base_model,
            tokenizer=tokenizer,
            src_lang="vie_Latn",
            tgt_lang="eng_Latn",
            max_length=128,
            device=0 if torch.cuda.is_available() else -1,
        )
        translate_func = lambda x: translator(x)[0]["translation_text"]

    elif name == "FP8":
        base_model.half()
        translator = pipeline(
            "translation",
            model=base_model,
            tokenizer=tokenizer,
            src_lang="vie_Latn",
            tgt_lang="eng_Latn",
            max_length=128,
            device=0 if torch.cuda.is_available() else -1,
        )
        translate_func = lambda x: translator(x)[0]["translation_text"]

    elif name == "ONNX Runtime":
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_name, export=True)
        ort_translator = pipeline(
            "translation",
            model=ort_model,
            tokenizer=tokenizer,
            src_lang="vie_Latn",
            tgt_lang="eng_Latn",
            max_length=128,
            device=0 if torch.cuda.is_available() else -1,
        )
        translate_func = lambda x: ort_translator(x)[0]["translation_text"]

    elif name == "Direct Model Generation":

        def translate_func(text):
            inputs = tokenizer(text, return_tensors="pt")

            translated_tokens = base_model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=128,
            )

            return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[
                0
            ]

    start_time = time.time()
    translations = []

    for text in tqdm(test_df["kha"].dropna(), desc=f"Translating with {name}"):
        translations.append(translate_func(text))

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_sentence = total_time / len(test_df["kha"].dropna())
    sentences_per_second = len(test_df["kha"].dropna()) / total_time

    return {
        "Method": name,
        "Total Translation Time (s)": total_time,
        "Avg Time per Sentence (s)": avg_time_per_sentence,
        "Sentences per Second": sentences_per_second,
    }


results = []
methods = ["HuggingFace Pipeline", "FP8", "ONNX Runtime", "Direct Model Generation"]

for method in methods:
    results.append(benchmark_translation_method(method, method))

benchmark_df = pd.DataFrame(results)
print(benchmark_df)

benchmark_df.to_csv("translation_benchmark_results.csv", index=False)
