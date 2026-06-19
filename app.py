from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
CORS(app)

def train_model():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'pregnancies': np.random.randint(0, 17, n_samples),
        'glucose': np.random.randint(70, 200, n_samples),
        'blood_pressure': np.random.randint(40, 130, n_samples),
        'skin_thickness': np.random.randint(10, 60, n_samples),
        'insulin': np.random.randint(15, 300, n_samples),
        'bmi': np.round(np.random.uniform(18, 50, n_samples), 1),
        'diabetes_pedigree': np.round(np.random.uniform(0.1, 2.5, n_samples), 3),
        'age': np.random.randint(21, 81, n_samples),
    }
    df = pd.DataFrame(data)
    df['outcome'] = (
        (df['glucose'] > 140).astype(int) +
        (df['bmi'] > 30).astype(int) +
        (df['age'] > 45).astype(int) +
        (df['blood_pressure'] > 90).astype(int) +
        (df['diabetes_pedigree'] > 1.0).astype(int)
    )
    df['outcome'] = (df['outcome'] >= 2).astype(int)
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
else:
    model = train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['pregnancies'],
        data['glucose'],
        data['blood_pressure'],
        data['skin_thickness'],
        data['insulin'],
        data['bmi'],
        data['diabetes_pedigree'],
        data['age']
    ]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100
    return jsonify({
        'prediction': int(prediction),
        'probability': round(float(probability), 2),
        'result': 'High Risk of Diabetes' if prediction == 1 else 'Low Risk of Diabetes'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)