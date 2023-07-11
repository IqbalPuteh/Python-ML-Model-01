import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Create the dictionary
dictionary = {
    "apple": "1001",
    "banana": "1002",
    "orange": "1003"
}

# Data preprocessing and splitting
words = list(dictionary.keys())
coas = list(dictionary.values())

word_to_index = {word: i for i, word in enumerate(words)}
index_to_word = {i: word for i, word in enumerate(words)}

coa_to_index = {coa: i for i, coa in enumerate(coas)}
index_to_coa = {i: coa for i, coa in enumerate(coas)}

def preprocess_data(dictionary):
    num_words = len(words)
    num_coas = len(coas)
    data = []
    for word, coa in dictionary.items():
        word_vec = np.zeros(num_words)
        coa_vec = np.zeros(num_coas)
        word_vec[word_to_index[word]] = 1
        coa_vec[coa_to_index[coa]] = 1
        data.append((word_vec, coa_vec))
    return data

preprocessed_data = preprocess_data(dictionary)

X = np.array([data[0] for data in preprocessed_data])
y = np.array([data[1] for data in preprocessed_data])

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

train_ratio = 0.8
train_size = int(X.shape[0] * train_ratio)
x_train, x_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the neural network model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the preprocessed training data
print("Training the model...")
model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)

# Evaluate the trained model on the testing data
print("Evaluating the model...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Your existing code for making predictions on new word vectors
print("Making predictions on new word vectors...")
new_word_vectors = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # Example new word vectors
new_word_vectors = np.array(new_word_vectors)
predictions = model.predict(new_word_vectors)
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Word Vector: {new_word_vectors[i]}")
    print(f"Predicted COA Vector: {prediction}")
    print()
