"""This is a newer version of the BLEU score calculation script. It allows for multiple references for each English sentence. It uses the sacrebleu library for BLEU and CHRF++ score calculation, which is more robust and handles multiple references better than the NLTK implementation."""
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sacrebleu
import pandas as pd
from collections import defaultdict

bleu = sacrebleu.BLEU()

trans_df = pd.read_csv(
    "../../../outputs/translated_filtered_manual_corpus.csv",
)

# Group target sentences by the corresponding English sentence
grouped_translations = defaultdict(list)
for en, kha, translated_en_to_kha in zip(
    trans_df["en"], trans_df["kha"], trans_df["translated_en_to_kha"]
):
    grouped_translations[en].append(kha)

target_sentences = trans_df["kha"].to_list()[:]
translated_sentences = trans_df["translated_en_to_kha"].to_list()[:]
english_sentences = trans_df["en"].to_list()[:]

smooth_fn = SmoothingFunction().method1

bleu_scores = []
for en, translated in zip(english_sentences, translated_sentences):
    references = grouped_translations[en]
    bleu_scores.append(
        sentence_bleu(
            [references], translated, smoothing_function=smooth_fn
        )
    )

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score (NLTK, considering multiple references) = {average_bleu_score}")

# Prepare references for corpus BLEU
corpus_references = []
for en in english_sentences:
    corpus_references.append(grouped_translations[en])

corpus_bleu_score = corpus_bleu(
    corpus_references, translated_sentences, smoothing_function=smooth_fn
)
print(f"Corpus BLEU score (NLTK, considering multiple references) = {corpus_bleu_score}")

# Prepare data for sacrebleu (it handles multiple references)
sacrebleu_references = [grouped_translations[en] for en in english_sentences]

print(
    "Corpus BLEU score (sacrebleu, considering multiple references): ",
    bleu.corpus_score(translated_sentences, sacrebleu_references),
)

print(
    "Corpus CHRF++ score (sacrebleu, considering multiple references): ",
    sacrebleu.corpus_chrf(translated_sentences, sacrebleu_references, word_order=2),
)
