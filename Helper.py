import torchtext.data as data
import nltk 
import numpy as np
import torch

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
    for idx,token in enumerate(tokenized):
        if token in all_word:
            index = all_word.index(token)
            bag[index] = 1
    return bag

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc