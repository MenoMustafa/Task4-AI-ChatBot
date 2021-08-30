
pip install nltk
pip install tflearn 


import json
import tensorflow as tf
import tflearn
import numpy
import random
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemm = LancasterStemmer()

from google.colab import files
files.upload()

with open('intents.json') as json_data:
    intents = json.load(json_data)

intents

words = []
classes = []
docs = []
ignore = ['?']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        docs.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [stemm.stem(word.lower()) for word in words if word not in ignore]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(docs), "Documents")
print (len(classes), "Classes", classes)
print (len(words), "Unique stemmed words", words)

train = []
output = []
output_empty = [0] * len(classes)

for doc in docs:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemm.stem(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    train.append([bag, output_row])

random.shuffle(train)
train = numpy.array(train)

train_x = list(train[:,0])
train_y = list(train[:,1])


from tensorflow.python.framework import ops
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=16, show_metric=True)
model.save('model.tflearn')

import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

def clean_sentence(sentence):
    sentence_w = nltk.word_tokenize(sentence)
    sentence_w = [stemm.stem(word.lower()) for word in sentence_w]
    return sentence_w
  
def bag_words(sentence, words, show_details=False):
    sentence_w = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_w:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(numpy.array(bag))
  

ERROR_THRESHOLD = 0.30
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list
  
def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    return print(random.choice(i['responses']))

