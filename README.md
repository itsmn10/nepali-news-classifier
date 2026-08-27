# Nepali News Classifier

Classifies Nepali news articles into categories (business, entertainment, sports)
using two approaches: a TF-IDF + Logistic Regression baseline, and a fine-tuned
NepaliBERT model.

## Results

| Model | Accuracy | F1 (macro) |
|---|---|---|
| TF-IDF + Logistic Regression | 92% | 0.92 |
| Fine-tuned NepaliBERT | [fill in] | [fill in] |

## Key finding

Error analysis showed a meaningful share of misclassifications stem from label
ambiguity in the source dataset — e.g., stories about international politics
(Trump/US-China trade, Myanmar's Aung San Suu Kyi) are labeled "business" despite
containing little business-specific vocabulary. This suggests part of the ~8%
baseline error rate reflects dataset annotation limits rather than model failure.

## Dataset

`disisbig/nepali-news-dataset` (Kaggle), 3 categories used from the train/valid split.

## Setup

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

See `notebooks/nepali_news_classifier.ipynb` for the full pipeline.
