import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical



# Load and preprocess MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Build LeNet-5 model with configurable activation and dropout
def build_lenet5(activation='sigmoid', dropout_rate=0.0):
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(6, (5, 5), activation=activation, input_shape=(28, 28, 1), padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    # Second convolutional block
    model.add(Conv2D(16, (5, 5), activation=activation))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    model.add(Dense(84, activation=activation))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train and evaluate model
def train_model(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=10):
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test)
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return test_accuracy, training_time, history

# Experiment with different activation functions
def activation_experiment():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    activations = ['sigmoid', 'relu', 'leaky_relu', 'elu']
    results = {}
    
    for activation in activations:
        print(f"Training with {activation} activation...")
        
        model = build_lenet5(activation=activation)
        accuracy, training_time, _ = train_model(model, x_train, y_train, x_test, y_test)
        
        results[activation] = {
            'accuracy': accuracy,
            'training_time': training_time
        }
        
        print(f"{activation}: Accuracy={accuracy:.4f}, Time={training_time:.2f}s")
    
    return results

# Experiment with different dropout rates
def dropout_experiment():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    dropout_rates = [0.0, 0.2, 0.4, 0.5]
    results = {}
    
    for rate in dropout_rates:
        print(f"Training with dropout rate {rate}...")
        
        model = build_lenet5_dropout(dropout_rate=rate)
        accuracy, training_time, _ = train_model(model, x_train, y_train, x_test, y_test)
        
        results[rate] = {
            'accuracy': accuracy,
            'training_time': training_time
        }
        
        print(f"Dropout {rate}: Accuracy={accuracy:.4f}, Time={training_time:.2f}s")
    
    return results

# Build LeNet-5 with dropout
def build_lenet5_dropout(activation='relu', dropout_rate=0.0):
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(6, (5, 5), activation=activation, input_shape=(28, 28, 1), padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Second convolutional block
    model.add(Conv2D(16, (5, 5), activation=activation))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
    model.add(Dense(84, activation=activation))
    if dropout_rate > 0:
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Experiment with different batch sizes and GPU vs CPU
def batch_size_experiment():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    batch_sizes = [32, 64, 128, 256]
    devices = ['/GPU:0', '/CPU:0']
    results = {}
    
    for device in devices:
        results[device] = {}
        
        for batch_size in batch_sizes:
            print(f"Training with batch size {batch_size} on {device}...")
            
            # Set device
            with tf.device(device):
                model = build_lenet5(activation='relu')
                
                start_time = time.time()
                model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    epochs=5,  # Fewer epochs for this experiment
                    verbose=0
                )
                training_time = time.time() - start_time
                
                # Evaluate
                test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
                
                results[device][batch_size] = {
                    'accuracy': test_accuracy,
                    'training_time': training_time
                }
                
                print(f"Device {device}, Batch {batch_size}: Accuracy={test_accuracy:.4f}, Time={training_time:.2f}s")
    
    return results

# Main execution
if __name__ == "__main__":
    print("=== Activation Function Experiment ===")
    activation_results = activation_experiment()
    
    print("\n=== Dropout Regularization Experiment ===")
    dropout_results = dropout_experiment()
    
    print("\n=== Batch Size and Device Experiment ===")
    batch_results = batch_size_experiment()
    
    # Print summary
    print("\n=== SUMMARY ===")
    
    print("\nActivation Functions:")
    for activation, metrics in activation_results.items():
        print(f"{activation}: Accuracy={metrics['accuracy']:.4f}, Time={metrics['training_time']:.2f}s")
    
    print("\nDropout Rates:")
    for rate, metrics in dropout_results.items():
        print(f"Rate {rate}: Accuracy={metrics['accuracy']:.4f}, Time={metrics['training_time']:.2f}s")
    
    print("\nBatch Sizes and Devices:")
    for device, batches in batch_results.items():
        for batch_size, metrics in batches.items():
            print(f"Device {device}, Batch {batch_size}: Accuracy={metrics['accuracy']:.4f}, Time={metrics['training_time']:.2f}s")
