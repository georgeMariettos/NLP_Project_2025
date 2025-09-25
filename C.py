import benepar, nltk, spacy
from spacy.cli import download
from nltk.tree import Tree
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "textattack/roberta-base-CoLA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nltk.download('punkt', quiet=True)
download("en_core_web_sm")
benepar.download('benepar_en3')

original_sentence1 = "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
original_sentence2 = "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."

reconstructed_sentence1 = ""
reconstructed_sentence2 = ""

bart_sentence1 = "Today is the Dragon Boat Festival, and in Chinese culture, we celebrate it with everything safe and good in our lives."
pegasus_sentence1 = "The dragon boat festival is a celebration in our Chinese culture and we should all be happy."
humarin_sentence1 = "Our Chinese culture features a dragon boat festival today, designed to celebrate with all that is good and safe in our lives."

bart_sentence2 = "Anyway, I trust that the team, even if a little late and less communicative in the last few days, has really tried its best in terms of paper and cooperation."
pegasus_sentence2 = "I believe the team tried their best for paper and cooperation despite the recent delay and less communication."
humarin_sentence2 = "Despite experiencing some delays and less communication than in recent days, the team did well in terms of paper-based and collaborative issues."


with open("reconstructed1.txt", encoding="utf-8") as f:
    for line in f:
        reconstructed_sentence1 += line

with open("reconstructed2.txt", encoding="utf-8") as f:
    for line in f:
        reconstructed_sentence2 += line


def get_parser():
    nlp = spacy.load("en_core_web_sm")

    if "benepar" not in nlp.pipe_names:

        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    return nlp

parser = get_parser()


def print_constituency_tree(text, sentence_name):

    print(f"Constituency tree for {sentence_name}")

    doc = parser(text)

    for i, sent in enumerate(doc.sents, start=1):

        tree = Tree.fromstring(sent._.parse_string)

        print(f"\nSentence {i}: {sent.text}")

        tree.pretty_print()

def cola_score(sentence):

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    return torch.softmax(logits, dim=-1)[0, 1].item()

def analyze_sentence(text, sentence_name):
    print_constituency_tree(text, sentence_name)
    score = cola_score(text)
    print(f"CoLA score for {sentence_name}: {score:.4f}")


analyze_sentence(original_sentence1, "Original Sentence 1")
analyze_sentence(reconstructed_sentence1, "Reconstructed Sentence 1")
analyze_sentence(original_sentence2, "Original Sentence 2")
analyze_sentence(reconstructed_sentence2, "Reconstructed Sentence 2")

model_sentences1 = {
    "BART Result for Sentence 1": bart_sentence1,
    "PEGASUS Result for Sentence 1": pegasus_sentence1,
    "HUMARIN Result for Sentence 1": humarin_sentence1,
}

model_sentence2 = {
    "BART  Result for Sentence 2": bart_sentence2,
    "PEGASUS  Result for Sentence 2": pegasus_sentence2,
    "HUMARIN  Result for Sentence 2": humarin_sentence2,
}

for label, sent in model_sentences1.items():
    analyze_sentence(sent, label)

for label, sent in model_sentence2.items():
    analyze_sentence(sent, label)

