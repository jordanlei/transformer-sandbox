import torch

def get_batch(data, block_size = 50, batch_size  = 64):
    # get a random starting index for each sample in our batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])

    # predict the next character
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y