import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'message': [
        'Free entry in 2 a wkly comp to win FA Cup final tkts',
        'Hello Tom, how\'s work?',
        'URGENT! You have won a 1 week FREE membership',
        'Hi Peter, the meeting is scheduled for tomorrow',
        'WINNER!! As a valued network customer you have been selected',
        'Hi John, can you send me the report?',
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    return prediction

if __name__ == "__main__":
    # Test the model with a new message
    test_message = "Congratulations! You've won a free iPhone! Click here to claim"
    print(f"\nTesting with message: {test_message}")
    print(f"Prediction: {predict_spam(test_message)}") 