import torchtext.data as data
import nltk 

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

def Bags_of_words(tokenized):
    pass