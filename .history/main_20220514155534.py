from transformers import pipeline

import torch
import torch, nn.functional as f

classififer = pipeline("sentiment-analysis")
