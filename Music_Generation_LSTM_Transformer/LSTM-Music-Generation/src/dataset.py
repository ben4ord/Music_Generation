import torch
from torch.utils.data import Dataset


# Sliding window dataset to create input-target pairs for training
class MusicDataset(Dataset):
    def __init__(self, token_sequences, genre_ids, seq_len):
        self.seq_len = seq_len
        self.samples = []

        # Create samples using a sliding window approach
        for tokens, genre in zip(token_sequences, genre_ids):
            for i in range(0, len(tokens) - seq_len):
                input_seq = tokens[i:i+seq_len]
                target_seq = tokens[i+1:i+seq_len+1]
                self.samples.append((input_seq, target_seq, genre))

    # return the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Return input sequence, target sequence, and genre id as tensors
    def __getitem__(self, idx):
        input_seq, target_seq, genre = self.samples[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(genre, dtype=torch.long),
        )