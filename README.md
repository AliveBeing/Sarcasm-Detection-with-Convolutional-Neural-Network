# Sarcasm Detection with TensorFlow and Keras

This repository contains code to build and train a neural network model to classify headlines as sarcastic or not using TensorFlow and Keras.

## Project Description

This project demonstrates a basic implementation of a neural network for sarcasm detection using a dataset of headlines. The dataset consists of headlines labeled as sarcastic or non-sarcastic. The goal is to build a model that can accurately classify new headlines.

## Code Overview

### Importing Libraries
The necessary libraries such as TensorFlow, Keras, Pandas, Numpy, and Scikit-learn are imported.

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
```

### Loading Data
The dataset is loaded from a JSON file containing headlines and their corresponding labels.

```python
df = pd.read_json('/content/sample_data/Sarcasm_Headlines_Dataset.json', lines=True)
print(df.head())
```

### Data Preprocessing
The headlines are tokenized and converted into sequences. The sequences are then padded to ensure uniform input length.

```python
sentences = df['headline'].values
labels = df['is_sarcastic'].values

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
```

### Splitting Data
The data is split into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
```

### Model Creation and Compilation
A Sequential model is created with the following layers:

1. **Embedding Layer:** This layer converts the input integer sequences into dense vectors of fixed size. We use an embedding dimension of 16.

2. **Convolutional Layer:** This layer applies a 1D convolution with 128 filters and a kernel size of 5. It helps in extracting local features from the sequences.

3. **GlobalMaxPooling1D Layer:** This layer downsamples the input representation by taking the maximum value over the time dimension, effectively reducing the dimensionality.

4. **Dense Layer:** A fully connected layer with 24 neurons and ReLU activation function. This layer helps in learning complex representations.

5. **Dropout Layer:** This layer helps in regularization by randomly setting a fraction of input units to 0 during training to prevent overfitting.

6. **Output Layer:** A single neuron with sigmoid activation function, which outputs the probability of the headline being sarcastic.

```python
vocab_size = len(word_index) + 1
embedding_dim = 16

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
```

### Training the Model
The model is trained on the training data for 10 epochs using the Adam optimizer and binary cross-entropy loss.

```python
epochs = 10
batch_size = 32

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
```

### Evaluation and Prediction
The model's performance is evaluated on the test data, and sample predictions are made to demonstrate its capabilities.

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.2f}')

def predict_sarcasm(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return "Sarcastic" if prediction >= 0.5 else "Not Sarcastic"

sample_text = "The weather is absolutely wonderful today!"
print(predict_sarcasm(sample_text))
sample_text = "boehner just wants wife to listen, not come up with alternative debt-reduction ideas"
print(predict_sarcasm(sample_text))
```

## Conclusion

This project provides a basic implementation of a neural network for sarcasm detection using TensorFlow and Keras. The model can be further improved by experimenting with different architectures, hyperparameters, and additional data preprocessing techniques.

