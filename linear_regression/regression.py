import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Reproducibility ---
np.random.seed(101)
tf.random.set_seed(101)

def generate_samples(n_samples: int = 50000) -> tuple:
    x = np.linspace(0, 50, n_samples).astype(np.float32)
    y = np.linspace(0, 50, n_samples).astype(np.float32)
    x += np.random.uniform(-4, 4, n_samples).astype(np.float32)
    y += np.random.uniform(-4, 4, n_samples).astype(np.float32)
    return x, y

def plot_raw_data(x, y):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter plot of X and Y')
    plt.show()

def plot_fitted_line(x, y, W, b):
    plt.scatter(x, y, label="Original data")
    x_line = np.linspace(x.min(), x.max(), 200).astype(np.float32).reshape(-1, 1)
    y_line = (x_line @ W.numpy()) + b.numpy()
    plt.plot(x_line, y_line, 'r', label="Fitted line")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Linear Regression Fit (TF2 stable)")
    plt.legend()
    plt.show()

def train_linear_regression_model(x,y):
    # --- Convert to tensors ---
    X = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
    Y = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)

    # --- Parameters (small init near 0) ---
    W = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))
    b = tf.Variable(tf.zeros([1], dtype=tf.float32))

    # --- Hyperparams ---
    learning_rate = 0.01
    epochs = 1000
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)

    # --- Model ---
    def predict(x_tensor):
        return tf.matmul(x_tensor, W) + b

    # --- Loss (MSE) ---
    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # --- Training loop ---
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = predict(X)
            loss = loss_fn(Y, y_pred)

        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: cost={loss.numpy():.6f}, W={W.numpy().ravel()[0]:.6f}, b={b.numpy().ravel()[0]:.6f}")

    print("\nTraining complete")
    print(f"Final cost={loss.numpy():.6f}, W={W.numpy().ravel()[0]:.6f}, b={b.numpy().ravel()[0]:.6f}")
    return W, b

# --- Generate noisy data ---
x, y = generate_samples(50)

# --- Plot raw data ---
plot_raw_data(x, y)

# --- Train model ---
W, b = train_linear_regression_model(x, y)

# --- Plot fitted line ---
plot_fitted_line(x, y, W, b)