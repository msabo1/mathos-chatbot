import cgi
import json
import random
from http.server import (BaseHTTPRequestHandler, HTTPServer,
                         SimpleHTTPRequestHandler)
from urllib.parse import unquote, urlparse

import nltk
import torch

from common import BagOfWords, ignore_words, normalize
from model import ChatbotModel

device = torch.device('cpu')

with open('intents.json', 'r') as intents_data_file:
    intents = json.load(intents_data_file)

model_data_file_name = 'current_model_data'

model_data = torch.load(model_data_file_name)

bag_of_words = BagOfWords(model_data['words'])

model = ChatbotModel(model_data['input_size'], model_data['hidden_size'], model_data['output_size']).to(device)
model.load_state_dict(model_data['state'])
model.eval()

intent_tag_to_id = {}
id = 0

for intent in intents['intents']:
    intent_tag_to_id[intent['tag']] = id
    id += 1


class MyHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, content-type") 
        self.end_headers()

    def do_OPTIONS(self):           
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '../frontent/index.html'
        

    def do_POST(self):
        self._set_headers()
        print(self.headers.get('content-type'))
        content_len = int(self.headers.get('Content-Length'))
        query = unquote(self.rfile.read(content_len).decode(encoding='utf_8'))
        query = dict(qc.split("=") for qc in query.split("&"))
        print(query)
        question = query['question']
        question = [normalize(word) for word in nltk.word_tokenize(question) if word not in ignore_words]
        if query.get('context') != None:
            question.append(query['context'])
        bow = bag_of_words.generate(question)
        bow = bow.reshape(1, bow.shape[0])
        out = model(torch.from_numpy(bow).to(torch.float).to(device))
        probabilities = torch.softmax(out, dim=1)
        intent_id = torch.argmax(probabilities[0])
        intent = intents['intents'][intent_id.item()]
        print(probabilities[0][intent_id])
        if probabilities[0][intent_id] < 0.7 or ((intent.get('context_filter') != None and query.get('context') == None) or (intent.get('context_filter') != query.get('context') if intent.get('context_filter') != None else False)):
            self.wfile.write(json.dumps({'answer': 'Na žalost, ne razumijem vaše pitanje!', 'context': query.get('context')}).encode())
        else:
            if intent.get('parent'):
                intent = intents['intents'][intent_tag_to_id[intent['parent']]]
            self.wfile.write(json.dumps({'answer': random.choice(intent['answers']), 'context': intent.get('context_set')}).encode())
    

server = HTTPServer(('', 3000), MyHandler)
print('Started listening')
server.serve_forever()


        