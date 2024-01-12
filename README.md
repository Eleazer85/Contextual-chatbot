# Chatbot

## About this project
This is a project for contextual / rule base chatbot using pytorch. In this project I'm learning about simple contextual chatbot and hopefully if you see this you can either use my code or model. I'm more than happy to answer any question regarding this project. 

## Model Architecture 
The model I'm using is a simple Linear model with ReLU activation and 8 hidden layer. Here's the model code: 
```
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
```
The length of the `input_size` and `num_classes` depends from the dataset. In my case the length of the input is 82 using Dataset.json as the dataset (I have 2 dataset, intents.json & Dataset.json) and my output length / `num_classes` is 19, which is also using Dataset.json .

## Tutorial
If you're wondering where do I learn this. I learned it from youtube using this tutorial by Patrick Loeber:
* [Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial Part 1](https://www.youtube.com/watch?v=RpWeNzfSUHw&ab)
* [Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial Part 2](https://www.youtube.com/watch?v=8qwowmiXANQ)
* [Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial Part 3](https://www.youtube.com/watch?v=Da-iHgrmHYg)
* [Chat Bot With PyTorch - NLP And Deep Learning - Python Tutorial Part 4](https://www.youtube.com/watch?v=k1SzvvFtl4w)
