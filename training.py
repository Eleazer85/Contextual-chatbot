import json
import torchtext.data as data
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Helper
import nltk

Tokenizer = data.get_tokenizer('basic_english')
with open('intents.json', 'r') as f:
 intents = json.load(f)
 
all_words = []
tags = []
xy = []
Stemmer = nltk.stem.PorterStemmer()

# Define a regular expression pattern for punctuation
punctuation_pattern = ["?","!",",","."]

# loop through each sentence in our intents patterns
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = Tokenizer(pattern)
        xy.append((w,tag))
        # Filter out punctuation character elements ans stemming the word
        w = [Stemmer.stem(x) for x in w if x not in punctuation_pattern]
        all_words.extend(w)
     
# Sort and remove duplicate words
all_words = sorted(set(all_words))
x_train = []
y_train = []

for sentence , tag in xy:
    bag = Helper.Bags_of_words(sentence,all_words)
    x_train.append(bag)
    y_train.append(tags.index(tag))
    
x_train , y_train = np.array(x_train) , np.array(y_train)

class TextDataset(Dataset): 
    def __init__(self,BoW,label):
        self.bag = BoW 
        self.label = label
    def __len__(self):
        return len(self.bag)
    def __getitem__(self,idx):
        return self.bag[idx] , self.label[idx]

Train_Dataset = TextDataset(x_train,y_train)
Train_loader = DataLoader(Train_Dataset,batch_size = 4,shuffle = True)