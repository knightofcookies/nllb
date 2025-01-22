from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
# from sacrebleu import BLEU


# bleu = BLEU()

with open("../../datasets/manual_en.txt", "r", encoding="utf-8") as file:
    target_sentences = [line.strip() for line in file]

with open("../../datasets/translated_manual_kha.txt", "r", encoding="utf-8") as file:
    translated_sentences = [line.strip() for line in file]

# target_sentences = ["I went to visit my aunt yesterday."]
# translated_sentences = ["I got to visit my aunt yesterday."]

smooth_fn = SmoothingFunction().method1

bleu_scores = [
    sentence_bleu([target], translated, smoothing_function=smooth_fn)
    for target, translated in zip(target_sentences, translated_sentences)
]

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score = {average_bleu_score}")

# print(bleu.corpus_score(target_sentences, translated_sentences))
