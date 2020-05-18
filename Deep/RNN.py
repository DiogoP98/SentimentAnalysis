import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import random


class rnn(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        #output, _ = nn.utils.rnn.pad_packed_sequence(hidden)
        out = self.fc(hidden[-1])
        
        return out
