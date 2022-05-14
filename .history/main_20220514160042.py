from transformers import pipeline
from transformers import 
import torch
import torch.nn.functional as f
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis",model=model_name)
res = classifier("This is so great")
print(res)
