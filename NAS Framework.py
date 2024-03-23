import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Define the space of hyperparameters to search
space  = [Integer(1, 3, name='num_layers'),
          Integer(32, 128, name='units_per_layer'),
          Real(10**-5, 10**-1, "log-uniform", name='learning_rate')]

# Function to create a model given a set of hyperparameters
def create_model(num_layers, units_per_layer, learning_rate):
    model = Sequential([Flatten(input_shape=(28, 28))])
    for _ in range(num_layers):
        model.add(Dense(units_per_layer, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Objective function to minimize
@use_named_args(space)
def objective(**params):
    model = create_model(**params)
    model.fit(x_train, y_train, epochs=5, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    # Negative accuracy because gp_minimize seeks to minimize the objective
    return -accuracy

# Run Bayesian Optimization
res_gp = gp_minimize(objective, space, n_calls=15, random_state=0)

# Results
print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- num_layers=%d
- units_per_layer=%d
- learning_rate=%.6f""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2]))
