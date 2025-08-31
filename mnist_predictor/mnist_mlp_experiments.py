import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load and preprocess MNIST ----------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten images: 28x28 → 784
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

# ---------------- Build the model ----------------
hidden_units = 512
learning_rate = 0.5  # you’ll want to experiment with smaller values too!

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(hidden_units, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ---------------- Train ----------------
EPOCHS = 20
BATCH_SIZE = 100

history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    verbose=2)

# ---------------- Evaluate ----------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# ---------------- Misclassified examples ----------------
pred_probs = model.predict(x_test)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_test, axis=1)
mis_idx = np.where(pred_labels != true_labels)[0]

print(f"Number of misclassified samples: {len(mis_idx)}")

# Plot a few misclassified digits
plt.figure(figsize=(10, 10))
for i, idx in enumerate(mis_idx[:25]):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"T:{true_labels[idx]} P:{pred_labels[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
