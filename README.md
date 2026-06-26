# CheckList Control Dataset for Sentiment Analysis

A small control dataset that tests three behaviours a sentiment model should
get right but often doesn't, following the CheckList method of Ribeiro et al.
(2020), "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList".

## Subsets

| File | Test type | Property tested | Rows |
|------|-----------|-----------------|------|
| `negation.csv` | MFT | Does the model read negation? | 216 |
| `names.csv` | INV | Does swapping a name change the prediction? | 24 |
| `noise.csv` | DIR/INV | Does neutral filler change the prediction? | 24 |

`negation.csv` has gold labels. `names.csv` and `noise.csv` are unlabelled:
they test relationships between predictions, not correctness against a label.

## Generate

```
python generate.py
```

## Evaluate

```
pip install transformers torch
python evaluate.py
```

Downloads `distilbert-base-uncased-finetuned-sst-2-english` and prints a
failure rate per subset.
