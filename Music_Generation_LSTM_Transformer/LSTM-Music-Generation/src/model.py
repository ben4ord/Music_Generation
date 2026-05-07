import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        genre_count,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # Need to tokenize genre labels as well, so we can embed them
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.genre_embedding = nn.Embedding(genre_count, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim, # input size is the embedding dimension, since we are adding token and genre embeddings together
            hidden_dim, # hidden dimension of the LSTM
            num_layers=num_layers, 
            batch_first=True, # we want our input tensors to be in the shape (batch, seq_len, feature_dim)
            dropout=dropout
        )

        # Set up linear layer to project LSTM output to vocab size for token prediction
        self.fc = nn.Linear(hidden_dim, vocab_size)

    # Forward method takes in a batch of token sequences and corresponding genre labels, and outputs logits for the next token prediction at each time step
    def forward(self, tokens, genre_id, hidden=None):

        #tokens: (batch, seq_len)
        #genre_id: (batch,)
        

        token_emb = self.token_embedding(tokens) # (batch, seq_len, embedding_dim)
        genre_emb = self.genre_embedding(genre_id).unsqueeze(1) # (batch, 1, embedding_dim) - unsqueeze to add a sequence dimension so we can add it to the token embeddings

        x = token_emb + genre_emb  # (batch, seq_len, embedding_dim) add the genre embedding to each token embedding in the sequence to condition the model on the genre

        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)   # project the LSTM output at each time step to the vocabulary size to get logits for the next token prediction

        return logits, hidden