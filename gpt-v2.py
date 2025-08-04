#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

batch_size = 8
block_size = 512
num_epochs = 5000

# Option 1: Use actual dataset size (recommended for true epochs)
# batches_per_epoch = train_batches_actual  # Will be calculated below

# Option 2: Use fixed number (current approach - good for random sampling)
batches_per_epoch = 200  # Number of batches per epoch

# Option 3: Use a fraction of actual dataset
# batches_per_epoch = train_batches_actual // 2  # Half the dataset per epoch

learning_rate = 1e-4
eval_frequency = 50  # Evaluate every N batches
eval_batches = 50   # Number of batches to evaluate on
n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.2

print(device)


chars = ""
with open("vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)


string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [string_to_int[c] for c in s]
def decode(l): return ''.join([int_to_string[i] for i in l])


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "wizard_of_oz.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode(
                'utf-8', errors='ignore').replace('\r', '')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Function to calculate dataset size and batches per epoch
def calculate_dataset_info():
    """Calculate the actual number of batches in one epoch based on dataset size"""

    # Get file sizes
    import os
    train_file = "wizard_of_oz.txt"
    val_file = "val_split.txt"

    # Calculate file sizes in characters
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
        train_chars = len(train_text)

    with open(val_file, 'r', encoding='utf-8') as f:
        val_text = f.read()
        val_chars = len(val_text)

    # Calculate number of sequences we can extract
    # Each sequence is block_size tokens long
    # We need block_size + 1 characters to create input and target
    sequence_length = block_size + 1

    train_sequences = train_chars // sequence_length
    val_sequences = val_chars // sequence_length

    # Calculate batches per epoch
    train_batches_per_epoch = train_sequences // batch_size
    val_batches_per_epoch = val_sequences // batch_size

    print(f"Dataset Analysis:")
    print(f"  Training file: {train_file}")
    print(f"    Characters: {train_chars:,}")
    print(f"    Sequences: {train_sequences:,}")
    print(f"    Batches per epoch: {train_batches_per_epoch:,}")
    print(f"  Validation file: {val_file}")
    print(f"    Characters: {val_chars:,}")
    print(f"    Sequences: {val_sequences:,}")
    print(f"    Batches per epoch: {val_batches_per_epoch:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Block size: {block_size}")
    print(f"  Current batches_per_epoch setting: {batches_per_epoch}")

    return train_batches_per_epoch, val_batches_per_epoch


# Calculate actual dataset info
train_batches_actual, val_batches_actual = calculate_dataset_info()


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, F) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index


model = GPTLanguageModel(vocab_size)

m = model.to(device)

# Initialize TensorBoard writer
log_dir = f"runs/gpt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")


# In[6]:


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# total_steps = 0  # Keep track of total training steps across all epochs

# for epoch in range(num_epochs):
#     print(f"Starting Epoch {epoch + 1}/{num_epochs}")

#     for batch_idx in range(batches_per_epoch):
#         total_steps += 1

#         # Evaluate and log losses periodically
#         if total_steps % eval_frequency == 0:
#             losses = estimate_loss()
#             train_loss = losses['train']
#             val_loss = losses['val']
#             print(
#                 f"Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Step: {total_steps}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}")

#             # Log to TensorBoard
#             writer.add_scalar('Loss/Train', train_loss, total_steps)
#             writer.add_scalar('Loss/Validation', val_loss, total_steps)
#             writer.add_scalar('Learning_Rate', learning_rate, total_steps)
#             writer.add_scalar('Epoch', epoch + 1, total_steps)

#             # # Save model weights at every evaluation step
#             # checkpoint_path = f'model_checkpoint_epoch_{epoch + 1}_step_{total_steps}.pth'
#             # torch.save(model.state_dict(), checkpoint_path)
#             # print(f"Model saved at epoch {epoch + 1}, step {total_steps}: {checkpoint_path}")

#         # sample a batch of data
#         xb, yb = get_batch('train')

#         # evaluate the loss
#         logits, loss = model.forward(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#         # Log training loss every iteration
#         writer.add_scalar('Loss/Train_Step', loss.item(), total_steps)

#     # Save model at the end of each epoch
#     epoch_checkpoint_path = f'model_epoch_{epoch + 1}.pth'
#     torch.save(model.state_dict(), epoch_checkpoint_path)
#     print(f"Epoch {epoch + 1} completed. Model saved: {epoch_checkpoint_path}")

# # Save model weights at last step
# final_checkpoint_path = 'last.pth'
# torch.save(model.state_dict(), final_checkpoint_path)
# print(f"Training completed! Model saved: {final_checkpoint_path}")

# print(f"Final loss: {loss.item()}")

# # Close the TensorBoard writer
# writer.close()


model = GPTLanguageModel(vocab_size)  # Create model first
model.load_state_dict(torch.load('model_epoch_312.pth'))

prompt = 'Hello! Can you see me?'
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
generated_chars = decode(m.generate(
    context.unsqueeze(0), max_new_tokens=100)[0].tolist())
print(generated_chars)
