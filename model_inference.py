
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import json
from caption_net import CaptionNet
from beheaded_inception3 import beheaded_inception_v3
from collections import Counter
import torch
from tqdm import tqdm_notebook
from random import choice
from tqdm import tqdm
from IPython.display import clear_output
from torch import nn
import torch.nn.functional as F
import pickle 

with open('word_to_index.pickle', 'rb') as fp:
    word_to_index = pickle.load(fp)

n_tokens =10403

vocab = [k for k, v in word_to_index.items()]

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    max_len = max_len or max(map(len,sequences))
    
    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i,seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

from beheaded_inception3 import beheaded_inception_v3

inception = beheaded_inception_v3().eval()

def generate_caption(image, network, caption_prefix = ('#START#',), t=1, sample=True, max_len=100):
    network = network.cpu().eval()

    assert isinstance(image, np.ndarray) and np.max(image) <= 1\
           and np.min(image) >= 0 and image.shape[-1] == 3
    
    image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)
    
    vectors_8x8, vectors_neck, logits = inception(image[None])
    caption_prefix = list(caption_prefix)
    
    for _ in range(max_len):
        
        prefix_ix = as_matrix([caption_prefix])
        prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64)
        next_word_logits = network.forward(vectors_neck, prefix_ix)[0, -1]
        next_word_probs = F.softmax(next_word_logits, -1).detach().numpy()
        
        assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t) # apply temperature

        if sample:
            next_word = np.random.choice(vocab, p=next_word_probs) 
        else:
            next_word = vocab[np.argmax(next_word_probs)]

        caption_prefix.append(next_word)

        if next_word == '#END#':
            break

    return ' '.join(caption_prefix[1:-1])


restored_network = CaptionNet(n_tokens)
restored_network.load_state_dict(torch.load('result.bin', map_location=torch.device('cpu')))

#get image from url

url = input('Provide a picture url: ')

f_name = url.split('/')[-1]
import os
os.system(f'wget {url}')

img = plt.imread(f_name)
img = resize(img, (299, 299))

from PIL import Image


for i in range(5):
    print(generate_caption(img, restored_network, t=5.))

img = Image.open(f_name)
img.show()
