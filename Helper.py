import torchtext.data as data
import nltk 
import numpy as np
Tokenizer = data.get_tokenizer('basic_english')
Stemmer = nltk.stem.PorterStemmer()

def yield_tokens(sentence):
    """
    Torch text tokenizer automatically 
    turn all of the uppercase to lowercase
    """
    return Tokenizer(sentence)
    
def Stemmer_fn(word): 
    return [Stemmer.stem(token) for token in word]

def Bags_of_words(tokenized,all_word):
    bag = np.zeros(len(all_word))
    tokenized = Stemmer_fn(tokenized)
    for idx,token in enumerate(tokenized):
        if token in all_word:
            bag[idx] = 1
    return bag