
# Install important packages 
pip install nltk
pip install tflearn 

# Import libraries
import json         # Upload and read json file
import tensorflow as tf         #Build deep learning model
import tflearn
import numpy        # Store data
import random       # Shuffle the training set
import nltk         # For text analysis
nltk.download('punkt')      # Download the pre-trained tokenized model for English language
from nltk.stem.lancaster import LancasterStemmer
stemm = LancasterStemmer()

# Load training data
from google.colab import files
files.upload()      # Upload intents file

with open('intents.json') as json_data:     # Open json file
    intents = json.load(json_data)          # Save file in the intents object

# Initialize files.
words = []
classes = []
docs = []
ignore = ['?']      # Ignore "?" symbol
# Loop for each sentence
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)      # Tokenize each word
        words.extend(word)      
        docs.append((word, intent['tag']))      # Adding words to the documents
        if intent['tag'] not in classes:        
            classes.append(intent['tag'])       # Add classes to the class list
            
words = [stemm.stem(word.lower()) for word in words if word not in ignore]      # Stem and lower each word, remove the duplicate words too
words = sorted(list(set(words)))
# Sort tags
classes = sorted(list(set(classes)))

print (len(docs), "Documents")
print (len(classes), "Classes", classes)        # Intent classes
print (len(words), "Unique stemmed words", words)       # All words and vocabs

# Initialize training data
train = []
output = []
output_empty = [0] * len(classes)       # Initialize empty array for output

for doc in docs:
    bag = []        # Initialize bag of words
    pattern_words = doc[0]      # Tokenized words list for the pattern
    pattern_words = [stemm.stem(word.lower()) for word in pattern_words]
    # Create array for bag of words with 1
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # For each pattern, the output is " 1 " for the current class, and " 0 " for each class 
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    train.append([bag, output_row])

random.shuffle(train)       # Shuffle the features
train = numpy.array(train)      # Make numpy array

# create training set and split it into features and labels
train_x = list(train[:,0])      # Features- pattern
train_y = list(train[:,1])      # Labels- tags


from tensorflow.python.framework import ops
ops.reset_default_graph()       # Reset the global default graph

# Build deep neural network model
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define the model 
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Train and save the model
model.fit(train_x, train_y, n_epoch=1000, batch_size=16, show_metric=True)
model.save('model.tflearn')

import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

def clean_sentence(sentence):
    sentence_w = nltk.word_tokenize(sentence)       # Split words into arrays using tokenize algorithm
    sentence_w = [stemm.stem(word.lower()) for word in sentence_w]
    return sentence_w
  
def bag_words(sentence, words, show_details=False):
    sentence_w = clean_sentence(sentence)       # Tokenize pattern
    bag = [0]*len(words)  
    for sentence in sentence_w:
        for i,w in enumerate(words):
            if w == sentence: 
                bag[i] = 1      # If the current word in the vocab position, assign 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(numpy.array(bag))
  

ERROR_THRESHOLD = 0.30
def classify(sentence):
    results = model.predict([bag_words(sentence, words)])[0]
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

