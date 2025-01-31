from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sacrebleu
import pandas as pd

bleu = sacrebleu.BLEU()

trans_df = pd.read_csv(
    # "../../datasets/translated_samanantar_50k_filtered.tsv",
    "../../datasets/translated_manual_corpus.csv",
    engine="python",
    # sep="\t\t\t\t\t",
)

target_sentences = trans_df["en"].to_list()[20000:]
translated_sentences = trans_df["translated_kha"].to_list()[20000:]

smooth_fn = SmoothingFunction().method1

bleu_scores = [
    sentence_bleu([target], translated, smoothing_function=smooth_fn)
    for target, translated in zip(target_sentences, translated_sentences)
]

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score (NLTK) = {average_bleu_score}")

print(
    "Corpus BLEU score (sacrebleu): ",
    bleu.corpus_score(target_sentences, translated_sentences),
)

print(
    "Corpus CHRF++ score (sacrebleu): ",
    sacrebleu.corpus_chrf(target_sentences, translated_sentences, word_order=2),
)
