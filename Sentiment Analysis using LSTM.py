# Import required libraries
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences to have same length
max_length = 100
x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create LSTM model
model = Sequential()
model.add(Embedding(10000, 128, input_length=max_length))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# Evaluate model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Use model for predictions
predictions = model.predict(x_test)
predicted_classes = (predictions > 0.5).astype('int32')

# Evaluate predictions
accuracy = accuracy_score(y_test, predicted_classes)
print(f'Prediction accuracy: {accuracy:.2f}')
