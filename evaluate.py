"""
evaluate.py
Runs a pretrained sentiment model over the three control subsets and
reports a failure rate for each. Run this locally (it downloads a model
from HuggingFace the first time).
Run: python evaluate.py
"""

import csv
from transformers import pipeline

# Maps the model's POSITIVE / NEGATIVE output to our label space.
clf = pipeline("sentiment-analysis",
               model="distilbert-base-uncased-finetuned-sst-2-english")


def predict(text):
    out = clf(text)[0]
    return out["label"].lower(), out["score"]  # 'positive' or 'negative'


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --- negation: compare prediction to gold label -------------------------
def eval_negation():
    rows = read("negation.csv")
    fails = 0
    for r in rows:
        pred, _ = predict(r["text"])
        # gold "positive" includes neutral; we only flag clear sign errors
        if pred != r["label"]:
            fails += 1
    return fails / len(rows)


# --- names: every variant in a group should get the same prediction -----
def eval_names():
    rows = read("names.csv")
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(predict(r["text"])[0])
    fails = sum(1 for preds in groups.values() if len(set(preds)) > 1)
    return fails / len(groups)


# --- noise: base and perturbed should get the same prediction -----------
def eval_noise():
    rows = read("noise.csv")
    fails = 0
    for r in rows:
        if predict(r["base"])[0] != predict(r["perturbed"])[0]:
            fails += 1
    return fails / len(rows)


if __name__ == "__main__":
    print(f"Negation MFT failure rate: {eval_negation():.1%}")
    print(f"Names INV failure rate:    {eval_names():.1%}")
    print(f"Noise INV failure rate:    {eval_noise():.1%}")
