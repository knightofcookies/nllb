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


df_train = pd.read_csv(
    "../../datasets/samanantar_4950k_filtered.tsv", sep="\t\t\t\t\t", engine="python"
)


model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# model = torch.nn.DataParallel(model)
model = model.cuda()

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


optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)


LANGS = [("en", "eng_Latn"), ("kha", "vie_Latn")]

df_index = 0


def get_batch_pairs(batch_size, data=df_train):
    global df_index
    if batch_size > len(data):
        raise Exception("Batch size too big!")
    if df_index + batch_size > len(data):
        data = data.sample(frac=1).reset_index(drop=True)
        df_index = 0
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[df_index]
        df_index += 1
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2


print(get_batch_pairs(1))


def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


batch_size = 8
max_length = 512
training_steps = 611397
losses = []
MODEL_SAVE_PATH = "./saved_model/fine_tuned_nllb"


model.train()
x, y, loss = None, None, None
cleanup()

tq = trange(len(losses), training_steps)
for i in tq:
    success = False
    xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
    while not success:
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(
                xx,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(
                yy,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            # -100 is a magic value ignored in the loss function
            # because we don't want the model to learn to predict padding ids
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            success = True

        except RuntimeError as e:  # usually, it is out-of-memory
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print("error", max(len(s) for s in xx + yy), e)
            continue

    if i % 1000 == 0:
        # each 1000 steps, I report average loss at these steps
        print(f"Step {i}, Average Loss: {np.mean(losses[-1000:])}")

    if i % 1000 == 0 and i > 0:
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)

cleanup()
