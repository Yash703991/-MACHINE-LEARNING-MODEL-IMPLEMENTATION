import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("="*50)
print("SPAM DETECTION PROGRAM")
print("="*50)

# Sample dataset
print("\nLoading sample dataset...")
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
print(f"Dataset loaded with {len(df)} messages")

# Split the data into training and testing sets
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} messages")
print(f"Testing set: {len(X_test)} messages")

# Create TF-IDF vectorizer
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Feature extraction complete")

# Train the model
print("\nTraining the Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("Model training complete")

# Make predictions
print("\nMaking predictions on test data...")
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("\nEvaluating model performance:")
print("-"*30)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("-"*30)

def predict_spam(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    return prediction

if __name__ == "__main__":
    print("\nTesting the model with new messages:")
    print("-"*30)
    
    test_messages = [
        "Congratulations! You've won a free iPhone! Click here to claim",
        "Hi Sarah, let's meet at the office tomorrow at 10 AM",
        "URGENT: Your account has been compromised, click here to reset",
        "The project report is ready for review"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        prediction = predict_spam(msg)
        print(f"\nMessage {i}: {msg}")
        print(f"Prediction: {prediction.upper()}")
        
    print("\nProgram execution complete!") 