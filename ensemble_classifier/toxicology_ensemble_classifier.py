import numpy as np
import pandas as pd
import deepchem as dc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import matplotlib.pyplot as plt

# ---------------------------
# 1) Set seeds and load data
# ---------------------------
np.random.seed(456)
tf.random.set_seed(456)

_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Restrict to first task
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# Normalize features
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
valid_X = scaler.transform(valid_X)
test_X = scaler.transform(test_X)

# CSV log
results = []


def log_result(method, params, val_acc, test_acc=None):
    results.append(
        {
            "method": method,
            "params": params,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
        }
    )


# ---------------------------
# 2) Random Forest Baseline
# ---------------------------
print("\n=== Random Forest Baseline ===")
rf_model = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=50,
    random_state=123,
    max_features="sqrt",
    min_samples_leaf=3,
)
rf_model.fit(train_X, train_y)

rf_results = {}
for split_name, X, y, w in [
    ("Train", train_X, train_y, train_w),
    ("Valid", valid_X, valid_y, valid_w),
    ("Test", test_X, test_y, test_w),
]:
    y_pred = rf_model.predict(X)
    score = accuracy_score(y, y_pred, sample_weight=w)
    rf_results[split_name] = score
    print(f"Weighted {split_name} Accuracy: {score:.4f}")

log_result(
    "RandomForest", {"n_estimators": 50}, rf_results["Valid"], rf_results["Test"]
)


# ---------------------------
# 3) Neural Network Evaluation
# ---------------------------
def eval_tox21_hyperparams(
    n_hidden=50,
    n_layers=1,
    learning_rate=0.001,
    dropout_prob=0.5,
    n_epochs=45,
    batch_size=100,
    weight_positives=True,
):
    model = keras.Sequential()
    model.add(layers.Input(shape=(train_X.shape[1],)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_hidden, activation="relu"))
        model.add(layers.Dropout(dropout_prob))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    # Compute sample weights (always start from DeepChem weights)
    sample_weights = train_w.copy()

    if weight_positives:
        # ratio = (# negatives / # positives)
        pos = np.sum(train_y == 1)
        neg = np.sum(train_y == 0)
        if pos > 0:
            weight_for_1 = neg / pos
        else:
            weight_for_1 = 1.0

        # Multiply positive examples' weights
        sample_weights = np.where(train_y == 1,
                                  sample_weights * weight_for_1,
                                  sample_weights)

    model.fit(
        train_X,
        train_y,
        sample_weight=sample_weights,
        validation_data=(valid_X, valid_y, valid_w),
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping],
    )

    valid_preds = (model.predict(valid_X, verbose=0).flatten() > 0.5).astype(int)
    val_score = accuracy_score(valid_y, valid_preds, sample_weight=valid_w)

    test_preds = (model.predict(test_X, verbose=0).flatten() > 0.5).astype(int)
    test_score = accuracy_score(test_y, test_preds, sample_weight=test_w)

    params = {
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "learning_rate": learning_rate,
        "dropout_prob": dropout_prob,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "weight_positives": weight_positives,
    }

    print(f"Params: {params} -> Val Acc={val_score:.4f}, Test Acc={test_score:.4f}")
    log_result("NeuralNet", params, val_score, test_score)
    return val_score, test_score

# ---------------------------
# 4) Random Search
# ---------------------------
param_space = {
    "n_hidden": [50, 100, 200, 400],
    "n_layers": [1, 2, 3],
    "learning_rate": [1e-2, 1e-3, 1e-4, 5e-3, 5e-4],
    "dropout_prob": [0.2, 0.3, 0.5],
    "batch_size": [50, 100, 200],
    "n_epochs": [20, 50, 100],
    "weight_positives": [True, False],
}


def sample_params(param_space):
    return {k: random.choice(v) for k, v in param_space.items()}


print("\n=== Random Search ===")
best_score, best_val_score, best_test_score, best_params = -1, None, None, None
repeats, n_trials = 3, 20
random_search_results = []

for trial in range(n_trials):
    params = sample_params(param_space)
    val_scores, test_scores = [], []
    for seed in range(repeats):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        val_score, test_score = eval_tox21_hyperparams(**params)
        val_scores.append(val_score)
        test_scores.append(test_score)
    avg_val_score, avg_test_score = np.mean(val_scores), np.mean(test_scores)
    random_search_results.append((avg_val_score, avg_test_score))
    print(
        f"Trial {trial + 1}/{n_trials} Avg Val Score={avg_val_score:.4f}, Avg Test Score={avg_test_score:.4f}"
    )
    if avg_val_score > best_score:
        best_score = avg_val_score
        best_val_score = avg_val_score
        best_test_score = avg_test_score
        best_params = params

print("\nBest Random Search Params:")
print(best_params)
print(
    f"Best Validation Accuracy: {best_val_score:.4f}, Test Accuracy: {best_test_score:.4f}"
)

# ---------------------------
# 5) Bayesian Optimization
# ---------------------------
print("\n=== Bayesian Optimization ===")
space = [
    Integer(50, 400, name="n_hidden"),
    Integer(1, 3, name="n_layers"),
    Real(1e-4, 0.5, prior="log-uniform", name="learning_rate"),
    Real(0.2, 0.8, name="dropout_prob"),
    Integer(50, 200, name="batch_size"),
    Integer(20, 100, name="n_epochs"),
    Categorical([True, False], name="weight_positives"),
]

bayes_results = []  # store (val_acc, test_acc) for each iteration


def objective(params):
    (
        n_hidden,
        n_layers,
        learning_rate,
        dropout_prob,
        batch_size,
        n_epochs,
        weight_positives,
    ) = params
    val_scores, test_scores = [], []
    for seed in range(3):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        val_score, test_score = eval_tox21_hyperparams(
            n_hidden=n_hidden,
            n_layers=n_layers,
            learning_rate=learning_rate,
            dropout_prob=dropout_prob,
            batch_size=batch_size,
            n_epochs=n_epochs,
            weight_positives=weight_positives,
        )
        val_scores.append(val_score)
        test_scores.append(test_score)
    avg_val, avg_test = np.mean(val_scores), np.mean(test_scores)
    bayes_results.append((avg_val, avg_test))
    return -avg_val


res = gp_minimize(objective, space, n_calls=30, random_state=42, verbose=True)

best_params_bayes = {
    "n_hidden": res.x[0],
    "n_layers": res.x[1],
    "learning_rate": res.x[2],
    "dropout_prob": res.x[3],
    "batch_size": res.x[4],
    "n_epochs": res.x[5],
    "weight_positives": res.x[6],
}
print("\nBest Bayesian Parameters:")
print(best_params_bayes)
print(f"Best Validation Accuracy: {-res.fun:.4f}")

# ---------------------------
# 6) Save results
# ---------------------------
df = pd.DataFrame(results)
df.to_csv("tox21_results.csv", index=False)
print("\nSaved results to tox21_results.csv")

# ---------------------------
# 7) Plot Random vs Bayesian (all iterations)
# ---------------------------
plt.figure(figsize=(8, 6))

# Random Search
rs_vals, rs_tests = zip(*random_search_results)
plt.scatter(rs_vals, rs_tests, label="Random Search", color="blue", alpha=0.6)

# Bayesian Optimization
bayes_vals, bayes_tests = zip(*bayes_results)
plt.scatter(
    bayes_vals, bayes_tests, label="Bayesian Optimization", color="red", alpha=0.6
)

plt.plot([0, 1], [0, 1], "k--", alpha=0.3)  # y=x reference
plt.xlabel("Validation Accuracy")
plt.ylabel("Test Accuracy")
plt.title("Random Search vs Bayesian Optimization (all iterations)")
plt.legend()
plt.grid(True)
plt.show()
