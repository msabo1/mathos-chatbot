import json

import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import BagOfWords, ignore_words, normalize
from dataset import ChatbotDataset
from model import ChatbotModel

with open('intents.json', 'r') as data_file:
    intents = json.load(data_file)

words = []
tags = []
X_train = []
y_train = []
id = 0
for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        sentence = [normalize(word) for word in nltk.word_tokenize(pattern) if word not in ignore_words]
        words.extend(sentence)
        if intent.get('context_filter') != None:
            words.append(intent['context_filter'])
            sentence.append(intent['context_filter'])
        X_train.append(sentence)
        y_train.append(id)
    id += 1
words = list(set(words))
tags = list(set(tags))

bag_of_words = BagOfWords(words)

for i in range(len(X_train)):
    X_train[i] = bag_of_words.generate(X_train[i])

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=float)

num_epochs = 1500
batch_size = 8
lr = 0.001
input_size = X_train.shape[1]
output_size = len(tags)
hidden_size = 8

dataset = ChatbotDataset(X_train, y_train)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = ChatbotModel(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for X, y in data_loader:
        X = X.to(torch.float).to(device)
        y = y.to(torch.long).to(device)
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 200 == 0:
        print('Epoch: ' + str(epoch) + ' / ' + str(num_epochs) + ', Loss: ' + str(loss.item()))

data = {
    'state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'words': words,
    'tags': tags
}

data_file_name = 'current_model_data'
torch.save(data, data_file_name)

print('Training done, model saved!')
    


