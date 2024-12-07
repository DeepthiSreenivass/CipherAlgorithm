from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
import json
import joblib
import os
from urllib.parse import urlparse, parse_qs
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import entropy, chisquare

app = Flask(__name__)
global cipher
# Global variable to store the input data
global_input_data = None

# File paths for the JSON databases
db_user_path = os.path.join('db', 'users.json')

# Ensure the necessary folders and files exist
if not os.path.exists('db'):
    os.makedirs('db')

if not os.path.exists(db_user_path):
    with open(db_user_path, 'w') as db_file:
        json.dump([], db_file)

# Helper functions for loading and saving data
def load_users():
    with open(db_user_path, 'r') as db_file:
        return json.load(db_file)

def save_users(users):
    with open(db_user_path, 'w') as db_file:
        json.dump(users, db_file, indent=4)

# Routes for the web application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        users = load_users()
        user = next((u for u in users if u['email'] == email and u['password'] == password), None)
        
        if user:
            return jsonify({'success': True, 'message': 'Successfully logged in!', 'redirect': url_for('home')})
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'})
    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        users = load_users()
        if any(u['email'] == email for u in users):
            return jsonify({'success': False, 'error': 'Email already registered'})
        else:
            users.append({'email': email, 'password': password})
            save_users(users)
            return jsonify({'success': True, 'message': 'Successfully signed up!', 'redirect': url_for('signin')})
    return render_template('signup.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    global cipher
    if request.method == 'POST':
        input_text = request.form.get('cipher-text', '').strip()
        if not input_text:
            return jsonify({'success': False, 'error': 'No input provided'}), 400

        cipher=input_text
        return redirect(url_for('result'))  # Redirect to the results page
    return render_template('input.html')

@app.route('/results', methods=['GET'])
def result():
    url = request.url
    parsed_url = urlparse(url)
    # Extract query parameters
    query_params = parse_qs(parsed_url.query)
    global cipher  # Declare the global variable

    model_path = "models/final_neural.h5"
    scaler_path = "models/final_scaler.pkl"
    feature_names_path = "models/feature_names.pkl"
    
    # Load the model, scaler, and feature names
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    # Check if there's any input data available
    if not cipher:
        return jsonify({'success': False, 'error': 'No input data found'}), 400

    # Predict the encryption algorithm
    try:
        predicted_algorithm_index = predict_algorithm(cipher, model, scaler, feature_names)
        algorithm = ['3DES', 'DES', 'AES']
        cipher_text = algorithm[predicted_algorithm_index]
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    # Render the results page with both the prediction and the input
    return render_template('results.html', cipher_text=cipher_text, last_input=cipher)

# Feature extraction and prediction functions
def extract_features(encrypted_message):
    byte_values = [int(encrypted_message[i:i+2], 16) for i in range(0, len(encrypted_message), 2)]
    length = len(encrypted_message)
    byte_distribution = [byte_values.count(i) / len(byte_values) for i in range(256)]
    entropy_value = entropy(byte_distribution)
    chi_square_value = chisquare(byte_distribution).statistic

    if len(byte_values) > 1:
        correlations = [byte_values[i] * byte_values[i+1] for i in range(len(byte_values)-1)]
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
    else:
        mean_correlation = 0
        std_correlation = 0

    return length, entropy_value, std_correlation

def predict_algorithm(encrypted_message, model, scaler, feature_names):
    features = extract_features(encrypted_message)
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    
    # Initialize the tokenizer and prepare the input for the model
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([encrypted_message])  # Fit tokenizer on the input text
    encrypted_message_seq = tokenizer.texts_to_sequences([encrypted_message])
    encrypted_message_padded = pad_sequences(encrypted_message_seq, maxlen=100)
    
    # Ensure the input to the model is in the correct format
    prediction = model.predict([encrypted_message_padded, features_scaled])
    return np.argmax(prediction[0])

if __name__ == '__main__':
    print("Connecting to the database and initializing the application...")
    app.run(debug=True)
    print("Application successfully connected and running.")
