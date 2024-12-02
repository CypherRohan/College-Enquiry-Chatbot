import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load intents file
with open(r'C:\Users\rohan\Desktop\College\ML\intents1.json') as file:
    intents = json.load(file)

# Prepare data
patterns, tags = [], []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

# Split data
X_train, X_test, y_train, y_test = train_test_split(patterns, encoded_tags, test_size=0.2, random_state=42)

# Text preprocessing function
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])

# Preprocess training and testing data
X_train = [preprocess_text(text) for text in X_train]
X_test = [preprocess_text(text) for text in X_test]

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test) 

print(f'Shape of X_train_vectorized: {X_train_vectorized.shape}')
print(f'Shape of X_test_vectorized: {X_test_vectorized.shape}')

# Train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_vectorized, y_train)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)  
svm_model.fit(X_train_vectorized, y_train)

# Evaluate models
logistic_predictions = logistic_model.predict(X_test_vectorized)
nb_predictions = nb_model.predict(X_test_vectorized)
svm_predictions = svm_model.predict(X_test_vectorized)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f'Logistic Regression Model Accuracy: {logistic_accuracy:.2f}')
print(f'Naive Bayes Model Accuracy: {nb_accuracy:.2f}')
print(f'SVM Model Accuracy: {svm_accuracy:.2f}')

# Saving the components for later use
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic_model, f)
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
