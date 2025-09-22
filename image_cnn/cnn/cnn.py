import tensorflow as tf
import numpy as np
import math
import os
from time import time


# ------------------------------
# Data loading function
# ------------------------------
def load_cifar10():
    """Load CIFAR-10 dataset and normalize to [0,1]."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert to one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


# ------------------------------
# Model definition
# ------------------------------
def build_model():
    """Builds a simple CNN model for CIFAR-10."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model


# ------------------------------
# Training loop
# ------------------------------
def train_model(model, train_data, test_data,
                batch_size=128, epochs=60,
                save_path="./checkpoints/cifar10"):
    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=3)

    best_acc = 0.0

    for epoch in range(epochs):
        start_time = time()
        train_loss.reset_state()
        train_accuracy.reset_state()

        # Training loop
        for batch_x, batch_y in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch_x, training=True)
                loss = loss_fn(batch_y, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.update_state(loss)
            train_accuracy.update_state(batch_y, predictions)

        # Validation
        test_accuracy.reset_state()
        for batch_x, batch_y in test_dataset:
            preds = model(batch_x, training=False)
            test_accuracy.update_state(batch_y, preds)

        duration = time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Loss: {train_loss.result():.4f}, "
              f"Train Acc: {train_accuracy.result():.4f}, "
              f"Test Acc: {test_accuracy.result():.4f}, "
              f"Time: {duration:.2f}s")

        # Save best model
        if test_accuracy.result() > best_acc:
            best_acc = test_accuracy.result()
            manager.save()
            print(f"New best accuracy: {best_acc:.4f}, checkpoint saved.")

    print(f"Training complete. Best Test Accuracy: {best_acc:.4f}")


# ------------------------------
# Evaluation function
# ------------------------------
def evaluate_model(model, test_data, batch_size=128):
    (x_test, y_test) = test_data
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
    for batch_x, batch_y in test_dataset:
        preds = model(batch_x, training=False)
        test_accuracy.update_state(batch_y, preds)

    print(f"Final Test Accuracy: {test_accuracy.result():.4f}")


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    (train_data, test_data) = load_cifar10()
    model = build_model()

    # Train
    train_model(model, train_data, test_data, epochs=10)

    # Evaluate
    evaluate_model(model, test_data)
