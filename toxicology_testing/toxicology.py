# Step 1: Load the Tox21 Dataset
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

np.random.seed(456)
tf.random.set_seed(456)

# Load dataset
_, (train, valid, test), _ = dc.molnet.load_tox21()

train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]

d = 1024
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(d,)),
        tf.keras.layers.Dense(n_hidden, activation="relu"),
        tf.keras.layers.Dropout(0.5),  # Step 6/7: dropout
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_X,
    train_y,
    validation_data=(valid_X, valid_y),
    batch_size=batch_size,
    epochs=n_epochs,
    verbose=1,
)

val_loss, val_acc = model.evaluate(valid_X, valid_y, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")

# Predictions
y_prob = model.predict(valid_X).ravel()
y_pred = (y_prob > 0.5).astype(int)

# Metrics
acc = accuracy_score(valid_y, y_pred)
f1 = f1_score(valid_y, y_pred, zero_division=0)
roc_auc = roc_auc_score(valid_y, y_prob)

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation F1-score: {f1:.4f}")
print(f"Validation ROC-AUC: {roc_auc:.4f}")

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
