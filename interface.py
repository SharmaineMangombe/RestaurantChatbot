from distutils.log import ERROR
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents= json.loads(open('Intents.json').read())

word = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model =load_model('chatbot-model.h5')

def clean_up_sentence(sentence):
    sentence_words =nltk.word_tokenize(sentence)
    sentence_words =[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(word)
    for w in sentence_words:
        for i in enumerate(word):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list =[]
    for  r in results:
        return_list.append({'intent': classes[r[0]], 'probality': str(r[1])})

    return return_list