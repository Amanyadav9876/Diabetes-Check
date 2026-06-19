from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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
    app.run(debug=True, port=5000)