[ReadMe.md](https://github.com/user-attachments/files/31510389/ReadMe.md)# Nepali News Classifier

Classifies Nepali news articles into categories (business, entertainment, sports)
using two approaches: a TF-IDF + Logistic Regression baseline, and a fine-tuned
NepaliBERT model.

## Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| TF-IDF + Logistic Regression | 92% | 0.92 |
| Fine-tuned NepaliBERT | **98%** | **0.98** |

## Key findings

**Error analysis on the baseline** showed that a meaningful share of
misclassifications stemmed from label ambiguity in the source dataset rather
than model failure. Articles about international politics (Trump/US-China trade
relations, Myanmar's Aung San Suu Kyi) are labeled "business" despite containing
little business-specific vocabulary — the dataset has no politics category, so
these get folded into whichever label is closest, which no vocabulary-based
model can reliably learn.

**Fine-tuning cut errors from 8% to 1.6%** (119 → 24 misclassified examples out
of 1,494), largely by resolving genre-level vocabulary confusion between
business and entertainment content that tripped up the TF-IDF baseline.
Notably, a handful of errors — such as a story about a celebrity's wealth status,
labeled entertainment but predicted business — persisted in **both** models.
This suggests a small residual of genuinely ambiguous content exists in the
dataset that remains difficult regardless of model sophistication, while the
bulk of the baseline's errors were addressable through better language
understanding.

## Dataset

[`disisbig/nepali-news-dataset`](https://www.kaggle.com/datasets/disisbig/nepali-news-dataset)
via Kaggle. Three categories (business, entertainment, sports) used from the
provided train/valid split. Article body text (`paras` column) was used as
classifier input rather than headlines, for a stronger text signal.

## Approach

1. **Preprocessing**: Unicode NFC normalization (critical for Devanagari script,
   which can encode visually identical characters multiple ways) and filtering
   to Devanagari + whitespace characters only.
2. **Baseline**: TF-IDF vectorization (unigrams + bigrams, 5000 features) +
   Logistic Regression with balanced class weights.
3. **Transformer**: Fine-tuned [`Shushant/nepaliBERT`](https://huggingface.co/Shushant/nepaliBERT),
   a BERT model pretrained on a Nepali news corpus, for 3 epochs.
4. **Error analysis**: Confusion matrix + manual inspection of misclassified
   examples for both models, to distinguish genuine model errors from dataset
   label noise.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

See `notebooks/nepali_news_classifier.ipynb` for the full pipeline, from data
loading through both models and error analysis.

## Project status

Built as a personal learning project to explore Nepali-language NLP, an
under-resourced area compared to English NLP tooling. Undergraduate
Electronics, Communication and Information Engineering student project.

## License

MIT


