import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve


class CIFAR10Trainer:
    def __init__(self, weight_decay=1e-4, dropout_rate=0.5):
        # Load CIFAR-10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)

        self.labels = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        # Build model
        self.model = self.build_model()

    def build_model(self):
        weight_decay = self.weight_decay
        dropout_rate = self.dropout_rate

        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomCrop(32, 32)
        ])

        model = keras.Sequential([
            data_augmentation,
            layers.Conv2D(32, (3,3), activation="relu", padding="same",
                          kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32,32,3)),
            layers.Conv2D(32, (3,3), activation="relu", padding="same",
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(dropout_rate),

            layers.Conv2D(64, (3,3), activation="relu", padding="same",
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Conv2D(64, (3,3), activation="relu", padding="same",
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(dropout_rate),

            layers.Flatten(),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dropout(dropout_rate),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, epochs=50, batch_size=128):
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=2
        )
        return history

    def evaluate(self, batch_size=128):
        # Predict probabilities
        y_pred_probs = self.model.predict(self.x_test, batch_size=batch_size)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("CIFAR-10 Confusion Matrix")
        plt.show()

        # Per-class accuracy
        class_acc = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class accuracy:")
        for i, acc in enumerate(class_acc):
            print(f"{self.labels[i]:<10}: {acc*100:.2f}%")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.labels))

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_probs.max(axis=1), n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0,1], [0,1], linestyle="--", label="Perfectly Calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    trainer = CIFAR10Trainer()
    trainer.train(epochs=50, batch_size=128)
    trainer.evaluate()
