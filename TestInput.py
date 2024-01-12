import torch 
import torch.nn as nn 
import Helper
import json 
import random 

device = "cuda" if torch.cuda.is_available() else "cpu"
with open('Dataset.json','r') as f:
    intents = json.load(f)
    
# Define a regular expression pattern for punctuation
punctuation_pattern = ["?","!",",","."]

tags = []
all_words = []
for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        w = Helper.Tokenizer(pattern)
        # Filter out punctuation character elements ans stemming the word
        w = [Helper.Stemmer.stem(x) for x in w if x not in punctuation_pattern]
        all_words.extend(w)  # Extend the list with processed words

all_words = sorted(list(set(all_words)))

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
    
model = NeuralNet(input_size=len(all_words),hidden_size=8,num_classes=len(tags)).to(device)
model.load_state_dict(torch.load('Text2nd.pth'))

while True: 
    Input = input("Ask the chatbot: ")
    Input = Helper.yield_tokens(Input)
    Input = Helper.Stemmer_fn(Input)
    bag = Helper.Bags_of_words(Input,all_words)
    bag = torch.tensor(bag).to(device=device,dtype=torch.float32)
    
    with torch.inference_mode():
        raw_preds = model(bag.unsqueeze(0))
        prediction = torch.softmax(raw_preds,dim=1).argmax(dim=1)
        
    total_response = len(intents["intents"][prediction.item()]["responses"])
    random_response = random.randint(0,total_response-1)
    print(f"Chatbot: {intents['intents'][prediction.item()]['responses'][random_response]}")