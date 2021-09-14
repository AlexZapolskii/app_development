
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import json
from collections import Counter
import torch
from tqdm import tqdm_notebook
from random import choice
from tqdm import tqdm
from IPython.display import clear_output
from torch import nn
import torch.nn.functional as F

n_tokens = 10403

class CaptionNet(nn.Module):
    
  def __init__(self, n_tokens=n_tokens, emb_size=128, lstm_units=256, cnn_feature_size=2048):
        """ A recurrent 'head' network for image captioning. . """
        super().__init__()
        
        # a layer that converts conv features to initial_h (h_0) and initial_c (c_0)
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)
        self.lstm_units = lstm_units
        # create embedding for input words. Use the parameters (e.g. emb_size).
        self.embedding = nn.Embedding(n_tokens, emb_size)

        # lstm: create a recurrent core of your network.
        self.LSTM = nn.LSTM(emb_size,lstm_units,num_layers = 1,batch_first=True)
            
        # create logits: linear layer that takes lstm hidden state as input and computes one number per token
        
        self.linear = nn.Linear(lstm_units, n_tokens)
        
  def forward(self, image_vectors, captions_ix):
        """ 
        Apply the network in training mode. 
        :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i]. 
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """

        self.LSTM.flatten_parameters()

        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)
        
        # compute embeddings for captions_ix
        x_emb = self.embedding(captions_ix)

        output, (hn, cn) = self.LSTM(x_emb, (initial_cell[None], initial_hid[None]))
        # compute logits from lstm_out
        output = self.linear(output)
        return output