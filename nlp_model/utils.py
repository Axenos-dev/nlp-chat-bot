import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize_sentence(sentence: str) -> list:
    return nltk.tokenize.word_tokenize(sentence)

def stem_word(word: str) -> str:
    return stemmer.stem(word, to_lowercase=True)

def bag_of_words(tokenized_sentence: list, vocabulary: list):
    stemmed_sentence = [ stem_word(word) for word in tokenized_sentence ]
    
    bag_of_words = np.zeros(len(vocabulary), dtype=np.float32)
    
    for index, word in enumerate(vocabulary):
        if word in stemmed_sentence:
            bag_of_words[index] = 1.
            
    return bag_of_words