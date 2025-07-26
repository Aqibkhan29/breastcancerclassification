import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
data = pd.read_csv('data.csv')
print("Dataset loaded successfully!")

# Step 2: Encode the target label
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Step 3: Drop unwanted columns
X = data.drop(columns=['id', 'diagnosis', 'Unnamed: 32'], errors='ignore')
y = data['diagnosis']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_single_sample(sample_values):
    import numpy as np
    sample_values = np.array(sample_values).reshape(1, -1)
    prediction = model.predict(sample_values)
    print("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")

# Pick one sample from test data to predict
sample = X_test.iloc[0].tolist()
predict_single_sample(sample)

import joblib

# Save model to a file
joblib.dump(model, "cancer_model.pkl")
print("Model saved as cancer_model.pkl")


