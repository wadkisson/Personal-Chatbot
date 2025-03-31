#THIS SHARD DATALOADER WAS COPIED FROM KARPATHY'S NANOGPT REPO. THIS IS NOT MY CODE.
import tiktoken
import numpy as np
import multiprocessing as mp
from datasets import load_dataset#pip install datasets
import os
import tqdm as tqdm#pip install tqdm


local_directory = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)
DATA_DIR = os.path.join(os.path.dirname(__file__), local_directory)
os.makedirs(DATA_DIR,exist_ok=True)
fw = load_dataset("HuggingFaceFW/fineweb-edu",name = remote_name,split = "train")
tokenizer = tiktoken.get_encoding("gpt2")
end_of_text_token = tokenizer._special_tokens['<|endoftext|>']

def tokenize_doc(document):
    tokens = [end_of_text_token]
    tokens.extend(tokenizer.encode_ordinary(document["text"]))
    toks_array = np.array(tokens)
    assert(0<=toks_array).all() and (toks_array < 2**16).all()
    formatted_toks = toks_array.astype(np.uint16)
    return formatted_toks

def write_file(filename,token_arr):
    with open(filename,"wb") as f:
        f.write(token_arr.tobytes())

nprocs = max(1,os.cpu_count()//2)

with mp.Pool(nprocs) as p:
    shard_idx = 0
    token_count = 0
    all_tokens = np.empty((shard_size,),dtype = np.uint16)
    token_counter = 0
    progress_bar = None
    for toks in p.imap(tokenize_doc,fw,chunksize = 16):
        if token_counter + len(toks) < shard_size:
            all_tokens[token_counter:token_counter+len(toks)] = toks
            token_counter += len(toks)
            if progress_bar is None:
                progress_bar = tqdm.tqdm(total=shard_size, unit="tokens", desc=f"Sharding {shard_idx}")
            progress_bar.update(len(toks))
        else:
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(DATA_DIR,f"fineweb_tok_{split}_{shard_idx}.bin")
            remainder = shard_size - token_counter
            progress_bar.update(remainder)
            all_tokens[token_count:token_count+remainder] = toks[:remainder]
            write_file(filename,all_tokens)
            shard_idx += 1
            progress_bar = None
            all_tokens[0:len(toks)-remainder] = toks[remainder:]
            token_counter = len(toks)-remainder
    if token_count != 0:
        split = "val" if shard_idx == 0 else "train"
        filename = os.path.join(DATA_DIR,f"fineweb_tok_{split}_{shard_idx}.bin")
        write_file(filename,all_tokens[:token_counter])
