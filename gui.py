import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import scrolledtext, ttk

# Load saved components
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load intents file
with open(r'C:\Users\rohan\Desktop\College\ML\intents1.json') as file:
    intents = json.load(file)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token.lower()) for token in tokens])

# Function to handle user input
def send():
    user_input = entry.get().strip() 
    if user_input.lower() == "exit":
        chat_window.insert(tk.END, "Chatbot: Goodbye!\n")
        root.quit()
    elif not user_input:
        chat_window.insert(tk.END, "Chatbot: Enter a valid question.\n")
    else:
        chat_window.insert(tk.END, "You: " + user_input + "\n")
        
        # Process and predict
        user_input_processed = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_processed])
        
        # Select model based on dropdown choice
        selected_model = model_selector.get()
        if selected_model == "Logistic Regression":
            predicted_tag = logistic_model.predict(user_input_vectorized)
        elif selected_model == "Naive Bayes":
            predicted_tag = nb_model.predict(user_input_vectorized)
        elif selected_model == "SVM":
            predicted_tag = svm_model.predict(user_input_vectorized)
        else:
            chat_window.insert(tk.END, "Chatbot: Please select a valid model.\n")
            return
        
        # Decode the predicted tag
        tag = label_encoder.inverse_transform(predicted_tag)[0]

        # Select a random response
        response = random.choice(
            [resp for intent in intents['intents'] if intent['tag'] == tag for resp in intent['responses']]
        )
        
        chat_window.config(state=tk.NORMAL)  
        chat_window.insert(tk.END, "Chatbot: " + response + "\n")
        chat_window.config(state=tk.DISABLED)  
        entry.delete(0, tk.END)  

# Function to update accuracy based on selected model
def update_accuracy(event):
    selected_model = model_selector.get()
    if selected_model == "Logistic Regression":
        accuracy_label.config(text="Accuracy: 69%")
    elif selected_model == "Naive Bayes":
        accuracy_label.config(text="Accuracy: 49%")
    elif selected_model == "SVM":
        accuracy_label.config(text="Accuracy: 81%")

# Create the main window
root = tk.Tk()
root.title("College Enquiry Chatbot")

# Create a scrolled text area for chat history
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_window.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_window.config(state=tk.DISABLED) 

# Create a dropdown for selecting the model
model_selector_label = tk.Label(root, text="Select Model:")
model_selector_label.pack(pady=5)

model_selector = ttk.Combobox(root, values=["Logistic Regression", "Naive Bayes", "SVM"], state="readonly")
model_selector.pack(pady=5)
model_selector.set("Logistic Regression")  # Default selection


model_selector.bind("<<ComboboxSelected>>", update_accuracy)


accuracy_label = tk.Label(root, text="Accuracy: 69%")  
accuracy_label.pack(pady=5)


entry = tk.Entry(root, width=80)
entry.pack(pady=10, padx=10)


send_button = tk.Button(root, text="Send", command=send)
send_button.pack(pady=10)


root.mainloop()
