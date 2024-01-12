import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Helper
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

with open('Dataset.json', 'r') as f:
    intents = json.load(f)
 
all_words = []
tags = []
xy = []

# Define a regular expression pattern for punctuation
punctuation_pattern = ["?","!",",","."]

# loop through each sentence in our intents patterns
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = Helper.Tokenizer(pattern)
        # Filter out punctuation character elements ans stemming the word
        w = [Helper.Stemmer.stem(x) for x in w if x not in punctuation_pattern]
        all_words.extend(w)
        xy.append((w,tag))
     
# Sort and remove duplicate words
all_words = sorted(set(all_words))
x_train = []
y_train = []

for sentence , tag in xy:
    bag = Helper.Bags_of_words(sentence,all_words)
    x_train.append(bag)
    y_train.append(tags.index(tag))
    
x_train , y_train = np.array(x_train) , np.array(y_train)

#Setup hyper parameter 
BATCH_SIZE = 8
LR = 0.001
HIDDEN_LAYER = 8
EPOCH = 2000

class TextDataset(Dataset): 
    def __init__(self,BoW,label):
        self.bag = BoW 
        self.label = label
    def __len__(self):
        return len(self.bag)
    def __getitem__(self,idx):
        return self.bag[idx] , self.label[idx]
        
Train_Dataset = TextDataset(x_train,y_train)
Train_loader = DataLoader(Train_Dataset,batch_size=BATCH_SIZE,shuffle = True)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,num_classes)
        )
    def forward(self,x):
        return self.layer(x)

print(all_words)
exit()
model = NeuralNet(input_size=len(x_train[0]) , hidden_size=HIDDEN_LAYER , num_classes=len(tags)).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm.tqdm(range(EPOCH)):
    for (sentence , label) in Train_loader:
        #forward pass 
        raw_pred = model(sentence.to(device=device,dtype=torch.float32))
        #calculate the loss
        loss = loss_fn(raw_pred,label.to(device=device,dtype=torch.long))
        y_pred=torch.softmax(raw_pred,dim=1).argmax(dim=1)
        accuracy = Helper.accuracy_fn(y_pred=y_pred,y_true=label.to(device,dtype=torch.long))
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        #update the weights
        optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1} \nloss {loss.item()} \naccuracy {accuracy}")
torch.save(model.state_dict(),"Text2nd.pth")