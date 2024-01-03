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
# loop through each sentence in our intents patterns
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = Tokenizer(pattern)
        all_words.extend(w)
        xy.append((w,tag))
     
# Define a regular expression pattern for punctuation
punctuation_pattern = ["?","!",",","."]
     
# Filter out non-string elements
print(f"{xy} \n\n\n")
print(f"{tag} \n\n\n")
print(f"{all_words} \n\n\n")
all_words = [Stemmer.stem(w) for w in all_words if w not in punctuation_pattern]
print(all_words)