import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import scrolledtext

# Load saved components
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
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
    user_input = entry.get()
    if user_input.lower() == "exit":
        chat_window.insert(tk.END, "Chatbot: Goodbye!\n")
        root.quit()
    else:
        chat_window.insert(tk.END, "You: " + user_input + "\n")
        
        # Process and predict
        user_input_processed = preprocess_text(user_input)
        user_input_vectorized = vectorizer.transform([user_input_processed])
        predicted_tag = model.predict(user_input_vectorized)
        tag = label_encoder.inverse_transform(predicted_tag)[0]

        # Select a random response
        response = random.choice(
            [resp for intent in intents['intents'] if intent['tag'] == tag for resp in intent['responses']]
        )
        
        chat_window.config(state=tk.NORMAL)  # Enable editing
        chat_window.insert(tk.END, "Chatbot: " + response + "\n")
        chat_window.config(state=tk.DISABLED)  # Disable editing
        entry.delete(0, tk.END)  # Clear input field

# Create the main window
root = tk.Tk()
root.title("Chatbot")

# Create a scrolled text area for chat history
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_window.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
chat_window.config(state=tk.DISABLED)  # Make it read-only

# Create an entry field for user input
entry = tk.Entry(root, width=80)
entry.pack(pady=10, padx=10)

# Create a send button
send_button = tk.Button(root, text="Send", command=send)
send_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()