from transformers import pipeline

import torch
import torch.nn.functional as f
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis")
res = classifier("This is so great")
print(res)
