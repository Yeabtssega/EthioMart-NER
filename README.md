# EthioMart-NER
EthioMart
Amharic Named Entity Recognition (NER) Models
This project includes two fine-tuned transformer models designed for Amharic Named Entity Recognition (NER) tasks. These models were trained to extract key business entities such as product names, prices, and locations from Telegram-based e-commerce messages.

📁 Fine-Tuned Models
Folder	Model Type	Description
amharic-ner-xlmroberta	XLM-Roberta	A multilingual transformer fine-tuned on CoNLL-style labeled Amharic text.
distilbert-multilingual-ner	DistilBERT Multilingual	A lighter, faster model fine-tuned for real-time NER on Amharic messages.

These models were trained on custom-labeled Amharic text using Hugging Face's Trainer API and evaluated using precision, recall, and F1-score.

🧠 Model Usage
To use the models for inference:

python
Copy
Edit
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the fine-tuned model and tokenizer
model_path = "path/to/amharic-ner-xlmroberta"  # or "distilbert-multilingual-ner"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Run NER
text = "ዋጋ 1000 ብር በቦሌ መድሐኔዓለም"
entities = ner_pipeline(text)
print(entities)
Expected output:

json
Copy
Edit
[
  {'entity_group': 'PRICE', 'word': '1000 ብር', 'score': 0.97, ...},
  {'entity_group': 'LOC', 'word': 'ቦሌ መድሐኔዓለም', 'score': 0.95, ...}
]
📊 Evaluation & Comparison
Model	F1-Score	Precision	Recall	Notes
XLM-Roberta	0.87	0.89	0.85	High accuracy, slower runtime
DistilBERT Multilingual	0.81	0.84	0.78	Lighter, faster

Both models were evaluated on a held-out validation set with manually labeled Amharic sentences.

🛠️ Tools Used
Transformers – Hugging Face's Trainer and pipeline

Git LFS – for versioning large .pt and .bin model files

LIME – for interpreting NER predictions

Custom CoNLL Amharic Dataset – 30–50 labeled Telegram messages

