import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/tickets.csv")

X = df["text"]
y = df["category"]

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(X)

model = LogisticRegression()

model.fit(X_tfidf, y)

new_ticket = ["Money was deducted but my order failed"]

new_ticket_tfidf = vectorizer.transform(new_ticket)

prediction = model.predict(new_ticket_tfidf)

print("Prediction:", prediction)