import datetime
import json
import pickle
import random
from json import JSONEncoder

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

class ResponseDto:
    def __init__(self, text):
        self.author = "user"
        # self.date = datetime.datetime.now()
        self.text = text

class ResponseDtoEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents_internal.json').read())

words = pickle.load(open('words.pickle', 'rb'))
classes = pickle.load(open('classes.pickle', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/messages', methods=['POST'])
@cross_origin()
def update_record():
    print(request)
    record = json.loads(request.data)
    messageBody_ = record["message"]
    print("request body: " + messageBody_)
    ints = predict_class(messageBody_)
    res = get_response(ints, intents)
    resp = ResponseDtoEncoder().encode(ResponseDto(res))
    print("response: " + res)
    return resp

app.run(debug=True)
