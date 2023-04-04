from accelerate import Accelerator
from datasets import load_dataset
import datasets
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2Model, AdamW
import torch
import numpy as np
import torch.nn.functional as F

accelerator = Accelerator()

block_size = 512
batch_size = 8
num_training_steps = 10


data = load_dataset('tiny_shakespeare')['test']['text'][0]
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
data = tokenizer.tokenize(data)
data = np.array(tokenizer.convert_tokens_to_ids(data))

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y


model = GPT2Model.from_pretrained('distilgpt2')
optimizer = AdamW(model.parameters(), lr=5e-5)


model, optimizer = accelerator.prepare(model, optimizer)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_training_steps):
    X, Y = get_batch()
    optimizer.zero_grad()
    device = X.device
    b, t = X.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(X)
    logits = outputs
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    # return logits, loss


    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    progress_bar.update(1)
