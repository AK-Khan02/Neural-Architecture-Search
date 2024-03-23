# Enhanced Neural Architecture Search (NAS) with Bayesian Optimization

## Objective

This project implements an Enhanced Neural Architecture Search (NAS) system designed to automatically find the optimal architecture for a neural network on a given dataset. Using Bayesian Optimization, the system efficiently searches through a predefined hyperparameter space, including the number of layers, units per layer, and learning rate, to maximize model performance.

## How It Works

The NAS system leverages Bayesian Optimization, a strategy that builds a probabilistic model of the function mapping from hyperparameter values to the objective evaluated on a validation set. It then uses this model to make informed decisions about where to sample next:

1. **Data Preprocessing**: Normalizes the MNIST dataset and prepares it for training.
2. **Model Creation**: Dynamically constructs neural network models based on specified hyperparameters.
3. **Bayesian Optimization**: Uses `scikit-optimize` to efficiently explore the hyperparameter space with the goal of minimizing the negative accuracy (maximizing accuracy).
4. **Evaluation**: Trains the neural network on the training set and evaluates its performance on the test set.

## Requirements

- Python 3.x
- TensorFlow (or TensorFlow 2.x)
- Keras (included with TensorFlow 2.x)
- scikit-optimize

## Installation

Ensure you have Python installed, then use `pip` to install the required libraries:

```bash
pip install tensorflow scikit-optimize
```

Note: If you're using TensorFlow 2.x, Keras is included as `tensorflow.keras`. Adjust import statements in the code accordingly if you're using TensorFlow 1.x.

## Usage

1. **Load Your Dataset**: The current implementation uses the MNIST dataset. You can modify the data loading section to use your dataset.
2. **Define Hyperparameter Space**: In the `space` variable, define the range and type of hyperparameters you want to explore.
3. **Run NAS**: Execute the script to start the NAS process. The system will output the performance of each architecture it evaluates and print the best architecture and its performance at the end.

```python
# To run the NAS, simply execute the script
python NAS with Bayesian Optimization.py
```

## Example Output

```
Best score=-0.9876
Best parameters:
- num_layers=2
- units_per_layer=64
- learning_rate=0.001234
```

The negative score represents the negative accuracy, as the optimization process minimizes the objective function.

---
