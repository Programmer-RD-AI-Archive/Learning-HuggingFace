from transformers import pipeline

import torch
import torch, nn.functional as f

classifier = pipeline("sentiment-analysis")
res = classifier("This is so great")
print(res)
