import re
import pandas as pd
from tqdm.auto import trange
import random
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
import gc
import torch
from copy import deepcopy

parallel_df = pd.read_csv(
    "../../datasets/samanantar_corpus.tsv", sep="\t\t\t\t\t", engine="python"
)

parallel_train = parallel_df[0:4950000].copy()
parallel_dev = parallel_df[4950000:4975000].copy()
parallel_test = parallel_df[4975000:].copy()

kha_mono = pd.read_csv(
    "../../datasets/news_kha.tsv", sep="\t\t\t\t\t", engine="python"
)["kha"].tolist()
eng_mono = pd.read_csv(
    "../../datasets/high_quality_english_sentences.tsv",
    sep="\t\t\t\t\t",
    engine="python",
)["en"].tolist()

# model_name = "facebook/nllb-200-distilled-600M"
model_name = "../../saved_model/cycle_loss_nllb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]


def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


replace_nonprint = get_non_printing_char_replacer(" ")


def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean


model.cuda()
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

parallel_index = 0
eng_mono_index = 0
kha_mono_index = 0


def get_batch_pairs(batch_size):
    global parallel_train
    global parallel_index
    if batch_size > len(parallel_train):
        raise Exception("Batch size too big!")
    if parallel_index + batch_size > len(parallel_train):
        parallel_train = parallel_train.sample(frac=1).reset_index(drop=True)
        parallel_index = 0
    xx, yy = [], []
    for _ in range(batch_size):
        item = parallel_train.iloc[parallel_index]
        parallel_index += 1
        xx.append(preproc(item["kha"]))
        yy.append(preproc(item["en"]))
    return xx, yy


def get_eng_mono_batch(batch_size):
    global eng_mono
    global eng_mono_index
    if batch_size > len(eng_mono):
        raise Exception("Batch size too big!")
    if eng_mono_index + batch_size > len(eng_mono):
        random.shuffle(eng_mono)
        eng_mono_index = 0
    xx = []
    for _ in range(batch_size):
        item = eng_mono[eng_mono_index]
        eng_mono_index += 1
        xx.append(preproc(item))
    return xx


def get_kha_mono_batch(batch_size):
    global kha_mono
    global kha_mono_index
    if batch_size > len(kha_mono):
        raise Exception("Batch size too big!")
    if kha_mono_index + batch_size > len(kha_mono):
        random.shuffle(kha_mono)
        kha_mono_index = 0
    xx = []
    for _ in range(batch_size):
        item = kha_mono[kha_mono_index]
        kha_mono_index += 1
        xx.append(preproc(item))
    return xx


def encode_texts(texts, tokenizer, src_lang, max_length):
    tokenizer.src_lang = src_lang
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)


def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


batch_size = 24
max_length = 128
training_steps = 51000
losses = []
MODEL_SAVE_PATH = "../../saved_model/cycle_loss_nllb"

lambda_p = 1.0
lambda_k = 1.0
lambda_e = 1.0

model.train()
loss = None

x_eng_pl = None
x_kha_pl = None
x_eng_mono = None
x_kha_mono = None
cleanup()

tq = trange(len(losses), training_steps)
for i in tq:
    try:
        kha_parallel, eng_parallel = get_batch_pairs(batch_size)
        x_kha_pl = encode_texts(kha_parallel, tokenizer, "vie_Latn", max_length)
        x_eng_pl = encode_texts(eng_parallel, tokenizer, "eng_Latn", max_length)

        kha_mono_batch = get_kha_mono_batch(batch_size)
        x_kha_mono = encode_texts(kha_mono_batch, tokenizer, "vie_Latn", max_length)

        eng_mono_batch = get_eng_mono_batch(batch_size)
        x_eng_mono = encode_texts(eng_mono_batch, tokenizer, "eng_Latn", max_length)

        kha_to_eng = model.generate(
            **x_kha_mono,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=max_length,
        )
        eng_to_kha = model.generate(
            **x_eng_mono,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("vie_Latn"),
            max_length=max_length,
        )

        kha_reconstructed = model.generate(
            input_ids=kha_to_eng,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("vie_Latn"),
            max_length=max_length,
        )
        eng_reconstructed = model.generate(
            input_ids=eng_to_kha,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=max_length,
        )

        original_kha_embeddings = model.get_encoder()(
            x_kha_mono.input_ids, attention_mask=x_kha_mono.attention_mask
        ).last_hidden_state.mean(dim=1)
        reconstructed_kha_embeddings = model.get_encoder()(
            kha_reconstructed, attention_mask=None
        ).last_hidden_state.mean(dim=1)

        original_eng_embeddings = model.get_encoder()(
            x_eng_mono.input_ids, attention_mask=x_eng_mono.attention_mask
        ).last_hidden_state.mean(dim=1)
        reconstructed_eng_embeddings = model.get_encoder()(
            eng_reconstructed, attention_mask=None
        ).last_hidden_state.mean(dim=1)

        kha_cycle_loss = torch.nn.functional.mse_loss(
            reconstructed_kha_embeddings, original_kha_embeddings
        )
        eng_cycle_loss = torch.nn.functional.mse_loss(
            reconstructed_eng_embeddings, original_eng_embeddings
        )

        y_eng_pl = deepcopy(x_eng_pl)
        y_kha_pl = deepcopy(x_kha_pl)
        y_eng_pl.input_ids[y_eng_pl.input_ids == tokenizer.pad_token_id] = -100
        y_kha_pl.input_ids[y_kha_pl.input_ids == tokenizer.pad_token_id] = -100
        parallel_loss = (
            model(
                **x_kha_pl,
                labels=y_eng_pl.input_ids,
            ).loss
            + model(
                **x_eng_pl,
                labels=y_kha_pl.input_ids,
            ).loss
        )

        total_loss = (
            lambda_p * parallel_loss
            + lambda_k * kha_cycle_loss
            + lambda_e * eng_cycle_loss
        )
        total_loss.backward()

        losses.append(total_loss.item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    except RuntimeError as e:
        optimizer.zero_grad(set_to_none=True)
        x_eng_pl = None
        x_kha_pl = None
        x_eng_mono = None
        x_kha_mono = None
        cleanup()
        print(f"Error: {e}")
        continue

    if i % 1000 == 0:
        print(f"Step {i}, Average Loss: {np.mean(losses[-1000:])}")

    if i % 1000 == 0 and i > 0:
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)

cleanup()
