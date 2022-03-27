import random
import json
import pickle
from tabnanny import verbose
from humanize import activate
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer
from sklearn import metrics

from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout
from tensorflow import keras

lemmatizer = WordNetLemmatizer()

intents= json.loads(open('intents.json').read())

words =[]
classes =[]
documents =[]
ignore_letters =['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['Patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
print(words)