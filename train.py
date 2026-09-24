# ============================================================
# AI CUSTOMER SUPPORT TICKET CLASSIFIER
# ============================================================
#
# What this program does:
#
# 1. Reads our customer-support dataset
# 2. Separates the messages (X) from their categories (y)
# 3. Splits the data into training and testing data
# 4. Converts text into numbers using TF-IDF
# 5. Trains a Logistic Regression model
# 6. Uses the model to predict unseen tickets
# 7. Checks how accurate the model is
#
# Overall flow:
#
# Customer text
#      ↓
# Train/Test Split
#      ↓
# TF-IDF (text → numbers)
#      ↓
# Logistic Regression
#      ↓
# Prediction
#      ↓
# Category
#
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd

# Pandas helps us read and work with our CSV dataset.
# "pd" is just a short name for pandas.


from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer converts text into numbers.
#
# Machine-learning models cannot directly understand:
# "My payment failed"
#
# So TF-IDF converts the text into numerical features
# that the ML model can work with.


from sklearn.linear_model import LogisticRegression

# LogisticRegression is the machine-learning algorithm
# we are using to classify the support tickets.
#
# It will learn patterns between:
#
# text features → category
#
# Example:
# payment-related words → payment_problem


from sklearn.model_selection import train_test_split

# train_test_split divides our dataset into two parts:
#
# Training data → used to teach the model
# Testing data  → used to check the model


from sklearn.metrics import accuracy_score

# accuracy_score compares:
#
# what the model predicted
#          VS
# the actual correct answers
#
# and gives us an accuracy value.

import joblib
# ------------------------------------------------------------
# 2. READ OUR DATASET
# ------------------------------------------------------------

df = pd.read_csv("data/tickets.csv")

# read_csv() reads our CSV file.
#
# Our project structure is:
#
# ai-support-ticket-classifier/
# │
# ├── train.py
# │
# └── data/
#     └── tickets.csv
#
# Therefore:
# "data/tickets.csv"
# means:
# go into the data folder and read tickets.csv.
#
# The result is stored in "df".
#
# df = DataFrame
#
# A DataFrame is basically a table of data.


# ------------------------------------------------------------
# 3. SEPARATE INPUT (X) AND ANSWER (y)
# ------------------------------------------------------------

X = df["text"]

# X contains the CUSTOMER'S MESSAGE.
#
# Example:
#
# "I forgot my password"
# "My payment failed"
# "The app keeps crashing"
#
# X is the INPUT that our model will learn from.


y = df["category"]

# y contains the CORRECT ANSWER for each message.
#
# Example:
#
# account_problem
# payment_problem
# technical_problem
#
# So:
#
# X = input
# y = correct output
#
# This is supervised learning because
# we are giving the model examples with known answers.


# ------------------------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# We don't want the model to learn from ALL our data.
#
# We divide the data into:
#
# X_train → messages used for training
# y_train → correct categories for those messages
#
# X_test → messages the model has NOT seen
# y_test → correct categories used to check predictions
#
#
# test_size=0.25 means:
#
# 25% of our dataset → testing
# 75% of our dataset → training
#
# With 12 tickets:
#
# approximately 9 → training
# approximately 3 → testing
#
#
# random_state=42
#
# The split is random.
# random_state makes sure we get the same split
# every time we run the program.
#
# 42 isn't special. It's just a fixed number we chose.


print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# ------------------------------------------------------------
# 5. CREATE THE TF-IDF CONVERTER
# ------------------------------------------------------------

vectorizer = TfidfVectorizer()

# We are creating a TF-IDF "machine".
#
# Its job is:
#
# TEXT
#   ↓
# TF-IDF
#   ↓
# NUMBERS
#
# Example:
#
# "My payment failed"
#          ↓
#       TF-IDF
#          ↓
# [0.0, 0.71, 0.42, ...]
#
# The exact numbers depend on our dataset.


# ------------------------------------------------------------
# 6. CONVERT TRAINING TEXT INTO NUMBERS
# ------------------------------------------------------------

X_train_tfidf = vectorizer.fit_transform(X_train)

# This does TWO things:
#
# 1. FIT
#    Look at the training messages and learn the vocabulary
#    and word statistics.
#
# 2. TRANSFORM
#    Convert the training messages into numerical vectors.
#
#
# Example:
#
# Training text:
#
# "I forgot my password"
# "My payment failed"
# "The app keeps crashing"
#
# TF-IDF learns words such as:
#
# password
# payment
# failed
# app
# crashing
# etc.
#
# Then it converts the sentences into numbers.
#
# IMPORTANT:
# We use fit_transform() ONLY on training data.
#
# We don't want the test data influencing what the
# vectorizer learns.


# ------------------------------------------------------------
# 7. CONVERT TEST TEXT INTO NUMBERS
# ------------------------------------------------------------

X_test_tfidf = vectorizer.transform(X_test)

# Notice that we use ONLY transform() here.
#
# We do NOT use:
#
# vectorizer.fit_transform(X_test)
#
# Why?
#
# Because the test data should remain unseen.
#
# The vectorizer has already learned its vocabulary
# from the training data.
#
# So now we simply use that SAME vocabulary to convert
# the test messages into numbers.
#
#
# Training:
#
# X_train
#    ↓
# fit_transform()
#    ↓
# learn + convert
#
#
# Testing:
#
# X_test
#    ↓
# transform()
#    ↓
# convert using what was already learned


# ------------------------------------------------------------
# 8. CREATE THE MACHINE-LEARNING MODEL
# ------------------------------------------------------------

model = LogisticRegression()

# We are creating a Logistic Regression model.
#
# At this moment, the model has NOT learned our
# customer-support categories yet.
#
# Think of it as an empty student.
#
# Next, we will teach it using our training data.


# ------------------------------------------------------------
# 9. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(X_train_tfidf, y_train)

# fit() means:
#
# "Learn from these examples."
#
# We give the model:
#
# X_train_tfidf
#     ↓
# numerical representation of customer messages
#
# y_train
#     ↓
# correct category for each message
#
#
# The model tries to learn patterns such as:
#
# password / login / account
#          ↓
# account_problem
#
# payment / card / charged
#          ↓
# payment_problem
#
# app / error / crashing
#          ↓
# technical_problem
#
# The model is NOT simply storing the sentences.
# It is learning a mathematical relationship between
# the numerical features and the categories.


print("Model trained successfully!")


# ------------------------------------------------------------
# 10. MAKE PREDICTIONS ON THE UNSEEN TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(X_test_tfidf)

# predict() asks our trained model:
#
# "What category do you think these test messages belong to?"
#
# Remember:
#
# X_test_tfidf = numerical versions of unseen messages
#
# The model looks at those numbers and predicts:
#
# account_problem
# payment_problem
# technical_problem
# etc.
#
# The predictions are stored in:
#
# y_pred


# ------------------------------------------------------------
# 11. CHECK HOW ACCURATE THE MODEL IS
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

# y_test contains the ACTUAL answers.
#
# y_pred contains the MODEL'S predictions.
#
# accuracy_score() compares them.
#
# Example:
#
# Actual:     payment_problem
# Predicted:  payment_problem
#              ↓
#            CORRECT
#
#
# If:
#
# 3 out of 3 predictions are correct:
#
# accuracy = 1.0
#
# which means:
# 100% of those test examples were classified correctly.


print("Predictions:", y_pred)
print("Actual answers:", y_test.to_numpy())
print("Accuracy:", accuracy)



joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(model, "model.pkl")

print("Vectorizer saved!")
print("Model saved!")


# ============================================================
# END
# ============================================================
#
# Our complete ML pipeline is now:
#
# CSV DATA
#    ↓
# X and y
#    ↓
# Train/Test Split
#    ↓
# TF-IDF
#    ↓
# Text becomes numbers
#    ↓
# Logistic Regression
#    ↓
# Model learns patterns
#    ↓
# Predict unseen messages
#    ↓
# Check accuracy
#
# NEXT:
# We will improve the dataset and then turn this ML model
# into a FastAPI endpoint so that we can send a customer
# message through an API and receive the predicted category
# as JSON.
# ============================================================