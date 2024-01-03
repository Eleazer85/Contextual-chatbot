import json
import torchtext.data as data
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
        # Filter out non-string elements
        w = [Stemmer.stem(x) for x in w if x not in punctuation_pattern]
        all_words.extend(w)
     
# Sort and remove duplicate words
all_words = sorted(set(all_words))
x_train = []
y_train = []

for sentence , tag in xy:
    bag = Helper.Bags_of_words(sentence,all_words)
    print(bag)