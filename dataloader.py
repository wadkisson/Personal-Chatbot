#Basic cleanup of the dataloader from this repo: https://github.com/karpathy/nanoGPT
import tiktoken
import torch
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = tiktoken.get_encoding("gpt2")


TENSOR_CORES = 8  # For Jetson Orin Nano

if __name__ == '__main__':
    data = load_dataset("openwebtext", num_proc=TENSOR_CORES)

    train_and_test = data['train'].train_test_split(test_size=0.01, shuffle=True)


    def tokenize(data_set):
        ids = tokenizer.encode_ordinary(data_set['text']) 
        ids.append(tokenizer.eot_token)
        return {'ids': ids, 'len': len(ids)}


    toks = train_and_test.map(
        tokenize,
        remove_columns=['text'],
        desc="Tokenizing...",
        num_proc=TENSOR_CORES, 
    )

    for split, dset in toks.items():
        doc_len = np.sum(dset['len'], dtype=np.uint64)  
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')

    
        memory_arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(doc_len,))#take memory of our ram so it doesn't explode


        batches = 1024
        i = 0
        for batch_i in tqdm(range(batches), desc=f"writing {split}.bin"):
            batch = dset.shard(num_shards=batches, index=batch_i, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            memory_arr[i:i + len(arr_batch)] = arr_batch
            i += len(arr_batch)

        memory_arr.flush()
