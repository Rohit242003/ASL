from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Load the ASL prediction model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

# Labels dictionary for ASL
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
    6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
    12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
    18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
    24: 'y', 25: 'z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 42: 'space'
}

# Markov model for next-word prediction
first_possible_words = {}
second_possible_words = {}
transitions = {}


def expandDict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)


def get_next_probability(given_list):
    probability_dict = {}
    given_list_length = len(given_list)
    for item in given_list:
        probability_dict[item] = probability_dict.get(item, 0) + 1
    for key, value in probability_dict.items():
        probability_dict[key] = value / given_list_length
    return probability_dict


def trainMarkovModel():
    for line in open('markov_chain.txt'):
        tokens = line.rstrip().lower().split()
        tokens_length = len(tokens)
        for i in range(tokens_length):
            token = tokens[i]
            if i == 0:
                first_possible_words[token] = first_possible_words.get(token, 0) + 1
            else:
                prev_token = tokens[i - 1]
                if i == tokens_length - 1:
                    expandDict(transitions, (prev_token, token), 'END')
                if i == 1:
                    expandDict(second_possible_words, prev_token, token)
                else:
                    prev_prev_token = tokens[i - 2]
                    expandDict(transitions, (prev_prev_token, prev_token), token)

    first_possible_words_total = sum(first_possible_words.values())
    for key, value in first_possible_words.items():
        first_possible_words[key] = value / first_possible_words_total

    for prev_word, next_word_list in second_possible_words.items():
        second_possible_words[prev_word] = get_next_probability(next_word_list)

    for word_pair, next_word_list in transitions.items():
        transitions[word_pair] = get_next_probability(next_word_list)


def next_word(tpl):
    if isinstance(tpl, str):  # it is the first word of the string
        d = second_possible_words.get(tpl)
        if d is not None:
            return list(d.keys())
    if isinstance(tpl, tuple):  # incoming words are a combination of two words
        d = transitions.get(tpl)
        if d is None:
            return []
        return list(d.keys())
    return None


# Train the Markov model
trainMarkovModel()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.json['image']
    try:
        # Decode the base64 image data
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error('Failed to decode image data.')
            return jsonify({'prediction': '', 'suggestions': []})

        # Process the frame for ASL prediction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = ''
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])

                # Handle both string and integer predictions
                if isinstance(prediction[0], str):
                    predicted_character = prediction[0]
                else:
                    predicted_character = labels_dict.get(int(prediction[0]), '?')

        # Get next-word suggestions
        sentence = request.json.get('sentence', '').strip().lower()
        tokens = sentence.split()
        suggestions = []
        if len(tokens) == 1:
            suggestions = next_word(tokens[0])
        elif len(tokens) >= 2:
            suggestions = next_word((tokens[-2], tokens[-1]))

        return jsonify({'prediction': predicted_character, 'suggestions': suggestions})
    except Exception as e:
        logging.error(f'Error processing image data: {e}')
        return jsonify({'prediction': '', 'suggestions': []})


@app.route('/speak', methods=['POST'])
def speak():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'})

    try:
        logging.info(f"Speaking text: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()
        return jsonify({'status': 'success'})
    except Exception as e:
        logging.error(f"Error speaking text: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)