import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)  # One-hot encode labels

def create_model(layers, units):
    """Function to create a Keras model with varying layers and units."""
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(layers):
        model.add(Dense(units, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(layers, units):
    """Function to train and evaluate a model, returning its accuracy."""
    model = create_model(layers, units)
    model.fit(x_train, y_train, epochs=5, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

# Search space
layer_options = [1, 2, 3]
unit_options = [32, 64, 128]

best_acc = 0
best_architecture = None

# Search loop
for layers in layer_options:
    for units in unit_options:
        acc = evaluate_model(layers, units)
        print(f"Architecture: {layers} layers with {units} units - Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_architecture = (layers, units)

print(f"\nBest Architecture: {best_architecture[0]} layers with {best_architecture[1]} units - Best Accuracy: {best_acc:.4f}")
