from flask import Flask, render_template, request
import joblib
import json
import numpy as np
import requests
from textblob import TextBlob

app = Flask(__name__)

# Load models
vectorizer = joblib.load("models/vectorizer.pkl")
lr = joblib.load("models/logistic_regression.pkl")
nb = joblib.load("models/naive_bayes.pkl")
svm = joblib.load("models/svm.pkl")
rf = joblib.load("models/random_forest.pkl")

with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

feature_names = vectorizer.get_feature_names_out()


# 🔥 Trending News (WITH IMAGES)
def get_trending_news():
    try:
        url = "https://newsapi.org/v2/top-headlines?country=us&pageSize=6&apiKey=8c147858a6e347e3a2ae67c1e5513e45"
        response = requests.get(url)
        data = response.json()

        articles = data.get("articles", [])

        headlines = [
            {
                "title": article["title"],
                "url": article["url"],
                "image": article["urlToImage"] if article["urlToImage"] else ""
            }
            for article in articles
        ]

        return headlines

    except:
        return []


@app.route("/")
def home():
    headlines = get_trending_news()
    return render_template("index.html", headlines=headlines)


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"].strip()
    headlines = get_trending_news()

    if not news:
        return render_template("index.html", headlines=headlines)

    if len(news.split()) < 20:
        return render_template(
            "index.html",
            error="⚠️ Please enter a full news article (minimum 20 words).",
            headlines=headlines
        )

    # Vectorize
    vec = vectorizer.transform([news])
    vec_array = vec.toarray()[0]

    # Predictions
    lr_pred = lr.predict(vec)[0]
    nb_pred = nb.predict(vec)[0]
    svm_pred = svm.predict(vec)[0]
    rf_pred = rf.predict(vec)[0]

    # Ensemble Voting
    votes = [lr_pred, nb_pred, svm_pred, rf_pred]
    final_vote = max(set(votes), key=votes.count)

    if votes.count(0) == 2 and votes.count(1) == 2:
        final_vote = lr_pred

    label = "Real" if final_vote == 1 else "Fake"
    confidence = round(lr.predict_proba(vec)[0].max() * 100, 2)

    # Sentiment
    blob = TextBlob(news)
    polarity = round(blob.sentiment.polarity, 3)

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Explainability
    lr_scores = vec_array * lr.coef_[0]
    lr_top = np.argsort(lr_scores)[-5:][::-1]
    lr_words = [feature_names[i] for i in lr_top]

    class_log_probs = nb.feature_log_prob_[nb_pred]
    nb_scores = vec_array * class_log_probs
    non_zero_indices = np.where(vec_array > 0)[0]
    nb_word_scores = [(i, nb_scores[i]) for i in non_zero_indices]
    nb_word_scores = sorted(nb_word_scores, key=lambda x: x[1], reverse=True)
    nb_top = [i for i, score in nb_word_scores[:5]]
    nb_words = [feature_names[i] for i in nb_top]

    svm_scores = vec_array * svm.coef_[0]
    svm_top = np.argsort(svm_scores)[-5:][::-1]
    svm_words = [feature_names[i] for i in svm_top]

    rf_scores = vec_array * rf.feature_importances_
    non_zero_rf = np.where(vec_array > 0)[0]
    rf_word_scores = [(i, rf_scores[i]) for i in non_zero_rf]
    rf_word_scores = sorted(rf_word_scores, key=lambda x: x[1], reverse=True)
    rf_top = [i for i, score in rf_word_scores[:5]]
    rf_words = [feature_names[i] for i in rf_top]

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        lr_result=lr_pred,
        nb_result=nb_pred,
        svm_result=svm_pred,
        rf_result=rf_pred,
        metrics=metrics,
        lr_words=lr_words,
        nb_words=nb_words,
        svm_words=svm_words,
        rf_words=rf_words,
        sentiment=sentiment,
        polarity=polarity,
        headlines=headlines
    )


if __name__ == "__main__":
    app.run(debug=True)