import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Create sample diabetes dataset
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

# Create target based on risk factors
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")