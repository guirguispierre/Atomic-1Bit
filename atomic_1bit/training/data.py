import torch
from datasets import load_dataset
import tiktoken
import numpy as np

class TinyStoriesDataset:
    def __init__(self, split="train", context_length=256, subset_ratio=0.1):
        """
        split: 'train' or 'validation'
        context_length: sequence length
        subset_ratio: Fraction of dataset to use (TinyStories is huge, we use a slice)
        """
        print(f"Loading TinyStories ({split})...")
        # Ensure we stream or load a small slice to avoid memory blowup
        self.dataset = load_dataset("roneneldan/TinyStories", split=f"{split}[:{int(subset_ratio*100)}%]")
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        
        # Pre-tokenize? or tokenize on fly?
        # For speed on CPU, better to pre-tokenize a buffer.
        # But for startup speed, let's tokenize on fly for now.
        print(f"Loaded {len(self.dataset)} samples.")
        
    def get_batch(self, batch_size):
        # Pick random indices
        indices = np.random.randint(0, len(self.dataset), batch_size)
        rows = self.dataset.select(indices)
        
        batch_input_ids = []
        batch_targets = []
        
        for text in rows["text"]:
            # Encode
            # We need length = context_length + 1 (for input + target)
            ids = self.enc.encode(text)
            ids.append(self.enc.eot_token) # <|endoftext|>
            
            if len(ids) < self.context_length + 1:
                # Pad
                ids = ids + [self.enc.eot_token] * (self.context_length + 1 - len(ids))
            
            # Random crop if too long
            if len(ids) > self.context_length + 1:
                start = np.random.randint(0, len(ids) - self.context_length - 1)
                ids = ids[start : start + self.context_length + 1]
                
            batch_input_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y

if __name__ == "__main__":
    # Test
    ds = TinyStoriesDataset(subset_ratio=0.01) # 1% for test
    x, y = ds.get_batch(4)
    print("X:", x.shape)
    print("Y:", y.shape)
