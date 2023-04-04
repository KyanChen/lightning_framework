import time
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, GPT2Tokenizer
import numpy as np
import torch
from lightning.fabric import Fabric
torch.set_float32_matmul_precision('medium')


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, data_name='tiny_shakespeare', block_size=1024, batch_size=4, model_c=''):
        data = load_dataset(data_name)['test']['text'][0]
        tokenizer = GPT2Tokenizer.from_pretrained(model_c)
        data = tokenizer.tokenize(data)
        self.data = np.array(tokenizer.convert_tokens_to_ids(data))
        self.block_size = block_size
        self.batch_size = batch_size

    def __getitem__(self, index):
        index = torch.randint(len(self.data) - self.block_size, [])
        x = torch.from_numpy((self.data[index:index + self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[index + 1:index + 1 + self.block_size]).astype(np.int64))
        return x, y

    def __len__(self):
        return self.batch_size * 64


def main(fabric):
    model_c = '/data/kyanchen/.cache/huggingface/hub/models--gpt2-medium/snapshots/425b0cc90498ac177aa51ba07be26fc2fea6af9d'
    dataset_c = '/data1/kyanchen/.cache/huggingface/datasets/tiny_shakespeare/default/1.0.0/b5b13969f09fe8707337f6cb296314fbe06960bd9a868dca39e713e163d27b5e/dataset_info.json'
    compile = False

    block_size = 1024
    batch_size = 4
    num_iters = 20

    toy_dataset = ToyDataset(block_size=block_size, model_c=model_c)
    dataloader = torch.utils.data.DataLoader(toy_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    config_ = GPT2Config.from_pretrained(model_c)
    model = GPT2LMHeadModel(config_)
    optimizer = AdamW(model.parameters(), lr=5e-5)


    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
        print("compiling the model finished.")

    model = fabric.setup_module(model)
    optimizer = fabric.setup_optimizer(optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    # model, optimizer = fabric.setup(model, optimizer)

    total_time = 0
    for iter_num in range(num_iters):
        t0 = time.time()
        for X, Y in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=X, labels=Y)
            loss, logits = outputs['loss'], outputs['logits']
            # import pdb; pdb.set_trace()
            fabric.backward(loss)
            optimizer.step()
        dt = time.time() - t0
        total_time += dt
        print(f"iter {iter_num}: time {dt:.2f}s")
    print('all time: ', total_time/num_iters)

    # A100 BS=4 gpt2-medium num_iters=20 len_dataset=64*BS 4卡 num_workers=8 16-mixed
    # dpp: 5.48s


if __name__ == '__main__':
    fabric = Fabric(accelerator="auto", devices=4, strategy="ddp_spawn", precision="16-mixed")

    # Fabric(accelerator="auto", devices="auto", strategy="auto")
    # --accelerator [cpu|gpu|cuda|mps|tpu]
    #                                   The hardware accelerator to run on.
    #   --strategy [ddp|dp|deepspeed|fsdp|ddp_spawn]   Strategy for how to run across multiple
    #                                   devices.
    fabric.launch()
    main(fabric)
