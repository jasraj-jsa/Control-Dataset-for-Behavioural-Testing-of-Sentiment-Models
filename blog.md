# Testing What a Sentiment Model Actually Knows

Accuracy on a held-out test set doesn't tell you much about where a model breaks. If the test set was drawn from the same distribution as training data, the model can score 95% by learning surface patterns that fail the moment you change one word.

Ribeiro et al. (2020) make this point in [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118). They borrow the idea from software engineering: instead of one big benchmark, write unit tests for specific capabilities. Each test changes exactly one thing, so any failure has exactly one cause. They call the three test types MFT (Minimum Functionality Test, a labelled unit test), INV (Invariance, a change that shouldn't move the prediction), and DIR (Directional Expectation, a change whose effect is known in advance).

This dataset applies that idea to three things a sentiment model should handle but frequently doesn't.

---

## The three subsets

**Subset 1 — Negation (MFT, 216 rows)**

The question: does the model actually read "not", or does it just detect a positive word and stop?

Three negation patterns of increasing difficulty:

```
The {thing} was not {adjective}.
Although they tried, the {thing} was not {adjective}.
The {thing} was anything but {adjective}.
```

When the adjective is positive, the negated sentence is negative. When the adjective is negative ("not awful"), the sentence is positive or neutral. The label follows directly from the construction, so every row has a gold label.

| Text | Label |
|------|-------|
| The service was not great. | negative |
| Although they tried, the meal was not wonderful. | negative |
| The flight was anything but great. | negative |
| The product was not poor. | positive |

**Subset 2 — Names (INV, 24 rows)**

The question: does swapping a person name or place name change the prediction?

Names carry no sentiment, so they shouldn't. Each group of rows is the same sentence with a different name slotted in. The test isn't whether the prediction is correct — it's whether it stays constant across the group.

| Group | Text |
|-------|------|
| 0 | Sarah said the service was great. |
| 0 | Priya said the service was great. |
| 2 | The trip to Berlin was wonderful. |
| 2 | The trip to Mumbai was wonderful. |

**Subset 3 — Noise (DIR/INV, 24 rows)**

The question: does appending a neutral fragment change the prediction?

Each row pairs a clearly polarised base sentence with a perturbed version that has a neutral fragment appended — a handle, a URL, or a context-free fact. The sentiment doesn't change, so the prediction shouldn't either.

| Base | Perturbed |
|------|-----------|
| The service was great. | The service was great. @pi9QDK |
| The flight was terrible. | The flight was terrible. https://t.co/PWK1jb |
| The meal was wonderful. | The meal was wonderful. It is Tuesday. |

---

## How the dataset is generated

One script, `generate.py`, builds all three files from small hand-curated word lists: six positive adjectives, six negative adjectives, six neutral nouns, six person names, six place names, and four filler fragments.

The negation subset is the Cartesian product of nouns × adjectives × templates. The names subset fills four sentence templates with each name or place. The noise subset pairs each base sentence with each filler fragment. The lists are small on purpose — every adjective has an unambiguous polarity, so the gold labels are trustworthy. A borderline word would make the labels unreliable.

Dataset and code: https://github.com/jasraj-jsa/Control-Dataset-for-Behavioural-Testing-of-Sentiment-Models

---

## What a model actually does

Running DistilBERT fine-tuned on SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`) over all three subsets:

```
Negation MFT failure rate: 34.3%
Names INV failure rate:     0.0%
Noise INV failure rate:     0.0%
```

The model is robust on names and noise. On negation it fails a third of the time, which sits in the 19–54% range the paper reports across commercial and research models.

The aggregate hides where the failures actually are. Breaking it down by template:

| Template | Failure rate |
|----------|-------------|
| `was not {adj}` | 2.8% |
| `Although they tried, was not {adj}` | 0.0% |
| `anything but {adj}` | 100.0% |

The plain "not" and the buried "not" are both fine. "Anything but" breaks every single row. The model reads the positive adjective and ignores the construction entirely. That's a precise failure, and isolating it is exactly what a control dataset is for. A standard test set mixing all these patterns would average them into a number that looks acceptable.

---

**Paper:** Ribeiro et al. (2020), [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/abs/2005.04118), ACL 2020.

**Dataset and code:** https://github.com/jasraj-jsa/Control-Dataset-for-Behavioural-Testing-of-Sentiment-Models
