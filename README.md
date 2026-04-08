# Fake-news-detection
AI system to detect fake news using ML, explainability, and real-time insights.

# 📰 AI Fake News Detection System

## App Screenshot

<img width="1919" height="908" alt="Screenshot 1" src="https://github.com/user-attachments/assets/762b46ba-76de-4646-bfb6-7bc81bd14665" />

<img width="1882" height="913" alt="Screenshot 2" src="https://github.com/user-attachments/assets/85b34644-3f84-4db0-94a1-507a3375d4c4" />



An intelligent web application that detects whether a news article is **Real or Fake** using Machine Learning, Explainable AI, and real-time data integration.

---

## 🚀 Features

- 🔍 Fake News Detection using Multiple ML Models  
- 🧠 Ensemble Prediction System  
- 📊 Explainable AI (Keyword-based insights)  
- 😊 Sentiment Analysis (TextBlob)  
- 📈 Confidence Score Display  
- 🌍 Real-time Trending News (NewsAPI)  
- 🎨 Modern UI with interactive news cards  

---

## 🧠 Tech Stack

**Backend:** Flask  
**Machine Learning:** Scikit-learn  
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  

**NLP:**  
- TF-IDF Vectorization  
- TextBlob  

**Others:**  
- NumPy  
- Joblib  
- NewsAPI  

---

## ⚙️ How It Works

1. User inputs news article  
2. Text is converted using TF-IDF  
3. Multiple models predict (LR, NB, SVM, RF)  
4. Final result using ensemble voting  
5. System displays:
   - Prediction (Real/Fake)  
   - Confidence Score  
   - Sentiment Analysis  
   - Explainable Keywords  

---

## 🔍 Explainable AI

The system highlights important words that influenced each model’s decision, making predictions transparent and understandable.

---

## 🌍 Real-Time News

- Fetches trending news using NewsAPI  
- Displays them as clickable cards  
- Redirects users to full articles  

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 📂 Project Structure

fake-news-detection/
│
├── models/
│ ├── vectorizer.pkl
│ ├── logistic_regression.pkl
│ ├── naive_bayes.pkl
│ ├── svm.pkl
│ ├── random_forest.pkl
│ └── metrics.json
│
├── templates/
│ └── index.html
│
├── app.py
├── train_models.py
├── Fake.csv
├── True.csv
├── screenshot.png
└── README.md


---

## ⚠️ Limitations

- Depends on dataset quality  
- May misclassify short inputs  
- Does not verify source credibility  

---

## 🔮 Future Scope

- Deep Learning models (BERT, LSTM)  
- Source credibility analysis  
- Image/video fake news detection  
- Cloud deployment  

---

## 📚 References

- Fake News Detection – Shu et al. (2017)  
- LIAR Dataset – Wang (2017)  
- Fake News Survey – Zhou & Zafarani (2020)  

---

## 👨‍💻 Author

**Rupika Kumarasamy**

---

## ⭐ Tagline

> Verify before you amplify.
