import torch 
import torch.nn as nn 
import Helper
import json 
import random 

device = "cuda" if torch.cuda.is_available() else "cpu"
with open('intents.json','r') as f:
    intents = json.load(f)
tags = [intent['tag'] for intent in intents['intents']]

all_words = ["'", 'a', 'accept', 'anyon', 'are', 'bye', 'can', 'card', 'cash', 'credit', 'day', 'deliveri', 
             'do', 'doe', 'funni', 'get', 'good', 'goodby', 'have', 'hello', 'help', 'hey', 'hi', 'how', 'i', 
             'is', 'item', 'joke', 'kind', 'know', 'later', 'long', 'lot', 'mastercard', 'me', 'my', 'of', 
             'onli', 'pay', 'paypal', 's', 'see', 'sell', 'ship', 'someth', 'take', 'tell', 'thank', 'that', 
             'there', 'what', 'when', 'which', 'with', 'you']
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
    
model = NeuralNet(input_size=55,hidden_size=8,num_classes=7).to(device)
model.load_state_dict(torch.load('Text.pth'))

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