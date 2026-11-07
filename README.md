[README.md](https://github.com/user-attachments/files/34513475/README.md)

# Nepali News Classifier

Classifies Nepali (Devanagari) news text into **business**, **entertainment** or **sports** by fine-tuning NepaliBERT, compared against a TF-IDF + Logistic Regression baseline.

🚀 **[Live demo](https://huggingface.co/spaces/matinniraula/nepali-news-classifier-demo)** · 🤗 **[Model](https://huggingface.co/matinniraula/nepali-news-classifier)** · 📊 **[Dataset](https://www.kaggle.com/datasets/disisbig/nepali-news-dataset)**

## Results

| Model | Test accuracy |
|---|---|
| TF-IDF (1–2-grams, 5000 features) + Logistic Regression | ~92% |
| Fine-tuned NepaliBERT (`Shushant/nepaliBERT`) | ~98% |

Error analysis (confusion matrix and misclassified examples) is in the notebook. Several of the baseline's mistakes came from ambiguous or noisy labels rather than model failure.

## Approach

1. **Data:** Nepali news dataset from Kaggle (`disisbig/nepali-news-dataset`); train and validation splits combined, text column renamed to `text`.
2. **Preprocessing:** Unicode NFC normalization, keeping Devanagari characters only (`src/preprocess.py`).
3. **Baseline:** TF-IDF + Logistic Regression.
4. **Transformer:** Fine-tune `Shushant/nepaliBERT` with Hugging Face `Trainer` (max length 128, 3 epochs, learning rate 2e-5, batch size 16) on a Colab T4 GPU.
5. **Deployment:** Model published on the Hugging Face Hub and served as a Gradio app on a ZeroGPU Space.

## Try it

Use the [live demo](https://huggingface.co/spaces/matinniraula/nepali-news-classifier-demo), or load the model in Python:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "matinniraula/nepali-news-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

labels = ["business", "entertainment", "sports"]  # alphabetical order
text = "नेपाली राष्ट्रिय फुटबल टिमले आजको खेलमा जित हासिल गर्यो।"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=1)[0]
print({l: round(p.item(), 3) for l, p in zip(labels, probs)})
```

## Repository structure

```
notebook/        Colab notebook: data, baseline, fine-tuning, evaluation
src/preprocess.py   Nepali text preprocessing
requirements.txt
LICENSE           MIT license
```

## Run it yourself

1. Open the notebook in `notebook/` in Google Colab and select a **T4 GPU** runtime.
2. Add your Kaggle API token and Hugging Face token under Colab **Secrets** (never paste tokens into cells).
3. Run all cells.

## Limitations and next steps

- Only three categories; articles outside them are still forced into one of the three.
- Planned comparisons: TF-IDF + Linear SVM, multilingual BERT (`bert-base-multilingual-cased`) and XLM-R (`xlm-roberta-base`).
- The free ZeroGPU demo has a daily per-visitor GPU quota, so it may occasionally show a quota message.

## License

MIT

## Author

Matin Niroula, [GitHub](https://github.com/itsmn10) · [LinkedIn](https://www.linkedin.com/in/matin-niroula)
