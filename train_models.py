import pandas as pd
import joblib
import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load ISOT dataset files
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

# Combine and shuffle
df = pd.concat([fake, real], axis=0)
df = df.sample(frac=1, random_state=42)

# Use text column
X = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Models
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

nb = MultinomialNB()
nb.fit(X_train, y_train)

svm = LinearSVC()
svm.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Accuracy
metrics = {
    "Logistic Regression": accuracy_score(y_test, lr.predict(X_test)),
    "Naive Bayes": accuracy_score(y_test, nb.predict(X_test)),
    "SVM": accuracy_score(y_test, svm.predict(X_test)),
    "Random Forest": accuracy_score(y_test, rf.predict(X_test))
}

# Save models
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(nb, "models/naive_bayes.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(rf, "models/random_forest.pkl")

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nModels trained successfully on ISOT dataset.\n")
for model, acc in metrics.items():
    print(f"{model}: {round(acc * 100, 2)}%")