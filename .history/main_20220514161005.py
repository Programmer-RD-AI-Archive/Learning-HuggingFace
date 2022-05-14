from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as f

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier("This is so great")
print(res)
tokens = tokenizer.tokenize("This is so great")
token_ids = tokenizer("This is so great")
print(tokens, token_ids)

X_train = ["testgewsg", "grewgreger"]
batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    pass
