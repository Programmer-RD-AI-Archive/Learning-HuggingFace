from transformers import pipeline

import torch
import torch.nn.functional as f
model_name = "distilbert-base"
classifier = pipeline("sentiment-analysis")
res = classifier("This is so great")
print(res)
