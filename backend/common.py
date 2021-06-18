from collections import OrderedDict

import numpy as np

from stemmer.Croatian_stemmer import CroatianStemmer

ignore_words = ['.', ',', '?', '!']

stemmer = CroatianStemmer()

def normalize(word):
    accent_dict = OrderedDict()
    accent_dict['č'] = 'c'
    accent_dict['ć'] = 'c'
    accent_dict['dž'] = 'd'
    accent_dict['đ'] = 'd'
    accent_dict['š'] = 's'
    accent_dict['ž'] = 'z'
    word = word.lower()
    for accent in accent_dict:
        word = word.replace(accent, accent_dict[accent])
    return stemmer.stem(word)

class BagOfWords:
    def __init__(self, words):
        self.words = words
        self.word_index = {}
        for i, word in enumerate(words):
            self.word_index[word] = i

    def generate(self, sentence):
        bag_of_words = np.zeros(len(self.words), dtype=int)
        for word in sentence:
            if self.word_index.get(word) != None:
                bag_of_words[self.word_index[word]] += 1
        return bag_of_words


