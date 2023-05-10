# import numpy as np
# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
# data = 'hello world'
# data = tokenizer.tokenize(data)
# data = tokenizer.convert_tokens_to_ids(data)
#
# token = tokenizer.encode(data)
# print(data)
# print(token)
#
# import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

inputs = tokenizer(["Hello, my dog is cute", 'Hell'], padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits
