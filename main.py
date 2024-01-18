import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import reuters

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('reuters')

# Load the Reuters corpus
corpus = reuters.sents()
corpus_text = [' '.join(sent) for sent in corpus]

# Tokenize the sentences
tokenizer = Tokenizer()
for sentence in corpus_text:
    token_list = tokenizer.texts_to_sequences([sentence])[0]

total_words = len(tokenizer.word_index) + 1

# Create input sequences and their corresponding labels
input_sequences = []
for sentence in corpus:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to have a consistent length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_length-1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

# Function to generate the next word
def generate_next_word(seed_text, model, tokenizer, max_sequence_length, temperature=1.0):
    for _ in range(10):  # Adjust the number of words to predict
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
        probabilities = model.predict(token_list, verbose=0)[0]
        scaled_probabilities = np.log(probabilities) / temperature
        exp_probabilities = np.exp(scaled_probabilities)
        predicted = np.argmax(exp_probabilities / np.sum(exp_probabilities))
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# Test the model
seed_text = "The company"
predicted_text = generate_next_word(seed_text, model, tokenizer, max_sequence_length)
print(predicted_text)
