import torch
from datasets import load_dataset
import tiktoken
import numpy as np

class TinyStoriesDataset:
    def __init__(self, split="train", context_length=256, subset_ratio=0.1):
        print(f"Loading TinyStories ({split})...")
        self.dataset = load_dataset("roneneldan/TinyStories", split=f"{split}[:{int(subset_ratio*100)}%]")
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        print(f"Loaded {len(self.dataset)} samples.")
        
    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.dataset), batch_size)
        rows = self.dataset.select(indices)
        
        batch_input_ids = []
        batch_targets = []
        
        for text in rows["text"]:
            ids = self.enc.encode(text)
            ids.append(self.enc.eot_token)
            
            if len(ids) < self.context_length + 1:
                ids = ids + [self.enc.eot_token] * (self.context_length + 1 - len(ids))
            
            if len(ids) > self.context_length + 1:
                start = np.random.randint(0, len(ids) - self.context_length - 1)
                ids = ids[start : start + self.context_length + 1]
                
            batch_input_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y

class AlpacaDataset:
    def __init__(self, split="train", context_length=256, vocab_cap=None):
        print(f"Loading Alpaca Cleaned ({split})...")
        self.dataset = load_dataset("yahma/alpaca-cleaned", split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        self.vocab_cap = vocab_cap
        if vocab_cap:
            print(f"   [Pocket Mode] Vocab Capped to {vocab_cap} (Modulo)")
        print(f"Loaded {len(self.dataset)} samples.")
        
    def format_prompt(self, sample):
        text = f"### Instruction: {sample['instruction']}\n"
        if sample.get("input", ""):
            text += f"### Input: {sample['input']}\n"
        text += f"### Response: {sample['output']}"
        return text

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.dataset), batch_size)
        rows = self.dataset.select(indices)
        
        batch_input_ids = []
        batch_targets = []
        
        for i in range(len(rows)):
            row = rows[i]
            text = self.format_prompt(row)
            
            # Encode
            ids = self.enc.encode(text)
            
            # Pocket Mode Modulo
            if self.vocab_cap:
                ids = [t % self.vocab_cap for t in ids]
                
            ids.append(self.enc.eot_token)
            # Re-apply modulo to EOT if needed (EOT usually 50256)
            if self.vocab_cap:
                ids[-1] = ids[-1] % self.vocab_cap
            
            # Handle Length
            if len(ids) < self.context_length + 1:
                ids = ids + [self.enc.eot_token] * (self.context_length + 1 - len(ids))
            elif len(ids) > self.context_length + 1:
                 # Truncate from end for instructions usually, or random crop?
                 # For instruction tuning, we usually want the start.
                 ids = ids[:self.context_length + 1]
                
            batch_input_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y

if __name__ == "__main__":
    # Test
    # ds = TinyStoriesDataset(subset_ratio=0.01)
    ds = AlpacaDataset()
    x, y = ds.get_batch(4)
    print("X:", x.shape)
    print("Y:", y.shape)

class AlpacaDataset:
    def __init__(self, split="train", context_length=256):
        print(f"Loading Alpaca-Cleaned ({split})...")
        self.dataset = load_dataset("yahma/alpaca-cleaned", split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        print(f"Loaded {len(self.dataset)} samples.")
        
    def format_prompt(self, sample):
        # Alpaca Format
        # Instruction: ...
        # Input: ... (Optional)
        # Response: ...
        
        text = f"### Instruction: {sample['instruction']}\n"
        if sample.get("input", ""):
            text += f"### Input: {sample['input']}\n"
        text += f"### Response: {sample['output']}"
        return text

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.dataset), batch_size)
        rows = self.dataset.select(indices)
        
        batch_input_ids = []
        batch_targets = []
        
        for i in range(len(rows)):
            row = rows[i]
            text = self.format_prompt(row)
            
            ids = self.enc.encode(text)
            ids.append(self.enc.eot_token)
            
            # Handling length
            if len(ids) < self.context_length + 1:
                ids = ids + [self.enc.eot_token] * (self.context_length + 1 - len(ids))
            elif len(ids) > self.context_length + 1:
                # For instructions, we probably want the BEGINNING, not a random crop.
                # Unlike stories, instructions make no sense if cropped.
                # However, fitting strict context is hard.
                # Let's crop end.
                ids = ids[:self.context_length + 1]
                
            batch_input_ids.append(ids[:-1])
            batch_targets.append(ids[1:])
            
        x = torch.tensor(batch_input_ids, dtype=torch.long)
        y = torch.tensor(batch_targets, dtype=torch.long)
        return x, y
