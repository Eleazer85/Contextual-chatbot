import json
import torchtext.data as data
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
        w = [Stemmer.stem(w) for w in all_words if w not in punctuation_pattern]
        all_words.extend(w)
        xy.append((w,tag))
     
# Filter out non-string elements
print(all_words)