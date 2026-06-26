"""
generate.py
Builds a small CheckList-style control dataset for sentiment analysis.
Three subsets, one property each:
  1. negation (MFT)  - does the model actually read "not"?
  2. names    (INV)  - does swapping a name change the prediction?
  3. noise    (DIR)  - does adding neutral filler change the prediction?
Run: python generate.py
"""

import csv
import itertools
import random

random.seed(0)

# --- small word lists ---------------------------------------------------

pos_adj = ["great", "wonderful", "excellent", "fantastic", "lovely", "amazing"]
neg_adj = ["terrible", "awful", "horrible", "poor", "disappointing", "lousy"]
things  = ["service", "flight", "meal", "room", "trip", "product"]
names   = ["Sarah", "James", "Priya", "Wei", "Maria", "John"]
places  = ["Berlin", "Toronto", "Mumbai", "Lisbon", "Osaka", "Madrid"]
filler  = ["@pi9QDK", "https://t.co/PWK1jb", "It is Tuesday.", "My order id is 90218."]


# --- subset 1: negation (MFT, has gold labels) --------------------------
# Three negation patterns, easy to hard. A model that keys on the positive
# word and ignores the negation gets the whole positive-adjective half wrong.
#   plain:  "The {thing} was not {pos}."          -> negative
#   buried: "Although they tried, the {thing} was not {pos}."  -> negative
#   idiom:  "The {thing} was anything but {pos}."  -> negative
# The negative-adjective version of each is the control: it should read
# positive (or neutral), so the model can't pass just by always saying
# "negative" when it sees the word "not".

neg_templates = [
    "The {thing} was not {adj}.",
    "Although they tried, the {thing} was not {adj}.",
    "The {thing} was anything but {adj}.",
]

def make_negation():
    rows = []
    for t in neg_templates:
        for thing, adj in itertools.product(things, pos_adj):
            rows.append([t.format(thing=thing, adj=adj), "negative"])
        for thing, adj in itertools.product(things, neg_adj):
            rows.append([t.format(thing=thing, adj=adj), "positive"])
    return rows


# --- subset 2: names (INV, no label, group id ties variants together) ---
# Same sentence, only the name/place changes. Prediction should not move.

def make_names():
    rows = []
    gid = 0
    templates = [
        "{n} said the service was great.",
        "{n} thought the meal was awful.",
        "The trip to {p} was wonderful.",
        "The flight to {p} was terrible.",
    ]
    for t in templates:
        fill = names if "{n}" in t else places
        for val in fill:
            rows.append([gid, t.format(n=val, p=val)])
        gid += 1
    return rows


# --- subset 3: noise (DIR, no label, base vs perturbed) -----------------
# Append neutral filler. Sentiment direction should stay the same.

def make_noise():
    rows = []
    bases = [
        "The service was great.",
        "The flight was terrible.",
        "The meal was wonderful.",
        "The room was awful.",
        "The trip was excellent.",
        "The product was disappointing.",
    ]
    gid = 0
    for b in bases:
        for f in filler:
            rows.append([gid, b, f"{b} {f}"])
        gid += 1
    return rows


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    write("negation.csv", ["text", "label"], make_negation())
    write("names.csv",    ["group", "text"], make_names())
    write("noise.csv",    ["group", "base", "perturbed"], make_noise())
