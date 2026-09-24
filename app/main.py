from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# ------------------------------------------------------------
# 1. CREATE FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(
    title="AI Customer Support Ticket Classifier",
    description="API that classifies customer support tickets using Machine Learning",
    version="1.0"
)


# ------------------------------------------------------------
# 2. LOAD OUR TRAINED ML COMPONENTS
# ------------------------------------------------------------

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")


# ------------------------------------------------------------
# 3. DEFINE THE INPUT FORMAT
# ------------------------------------------------------------

class TicketRequest(BaseModel):
    message: str


# ------------------------------------------------------------
# 4. BASIC HOME ENDPOINT
# ------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Support Ticket Classifier is running!"
    }


# ------------------------------------------------------------
# 5. TICKET CLASSIFICATION ENDPOINT
# ------------------------------------------------------------

@app.post("/tickets/classify")
def classify_ticket(ticket: TicketRequest):

    # Get the customer's message
    message = ticket.message

    # Convert the message into TF-IDF numbers
    message_tfidf = vectorizer.transform([message])

    # Ask the trained ML model to predict the category
    prediction = model.predict(message_tfidf)

    # Get the predicted category
    category = prediction[0]

    # Return the result as JSON
    return {
        "message": message,
        "category": category
    }