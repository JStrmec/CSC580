#!/usr/bin/env python3
"""
Conditional DCGAN with hinge loss for CIFAR-10.

- Generator and Discriminator conditioned on class labels via embeddings.
- Hinge loss (linear D output).
- Per-class metrics: precision/recall/F1 and real/fake accuracies.
- Saves sample grids (one file per class) periodically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple, Any

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------- CONFIG ----------------
@dataclass
class CGANConfig:
    latent_dim: int = 100
    image_shape: Tuple[int, ...] = (32, 32, 3)
    lr: float = 2e-4
    beta_1: float = 0.5
    gen_filters: int = 128
    disc_filters: int = 64
    batch_size: int = 64
    n_critic: int = 1
    flip_label_prob: float = 0.03
    save_every: int = 10
    seed: int = 42
    image_folder: str = "generated_images"
    history_path: str = "training_history_cgan.png"
    classes: int = 10
    eval_samples_per_class: int = 200  # for per-class evaluation
    drop_real_prob: float = 0.0  # optional: sometimes drop real images (not used here)

# DCGAN init
kernel_init = tf.random_normal_initializer(0., 0.02)

# ---------------- MODELS (conditional) ----------------
def build_cifar_generator(cfg: CGANConfig) -> Model:
    # Condition by embedding label and concatenating to latent processing stream
    z_in = Input(shape=(cfg.latent_dim,), name="z")
    label_in = Input(shape=(), dtype="int32", name="label")  # scalar label

    # label embedding -> vector
    emb = layers.Embedding(cfg.classes, cfg.latent_dim, embeddings_initializer="uniform")(label_in)
    emb = layers.Flatten()(emb)

    # combine latent z and label embedding
    x = layers.Concatenate()([z_in, emb])  # shape (latent_dim*2)
    # project into spatial tensor
    x = layers.Dense(4 * 4 * cfg.gen_filters * 2, use_bias=False, kernel_initializer=kernel_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, cfg.gen_filters * 2))(x)

    x = layers.Conv2DTranspose(cfg.gen_filters, 4, strides=2, padding="same", use_bias=False, kernel_initializer=kernel_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(cfg.gen_filters // 2, 4, strides=2, padding="same", use_bias=False, kernel_initializer=kernel_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh", kernel_initializer=kernel_init)(x)

    return Model([z_in, label_in], x, name="generator_conditional")


def build_cifar_discriminator(cfg: CGANConfig) -> Model:
    # Condition by embedding label and concatenating as channels to image feature map
    img_in = Input(shape=cfg.image_shape, name="image")
    label_in = Input(shape=(), dtype="int32", name="label")

    # embed label and expand to spatial map (4x4), then tile to concatenate with conv features early
    emb = layers.Embedding(cfg.classes, 50, embeddings_initializer="uniform")(label_in)  # small embedding
    emb = layers.Dense(4 * 4 * 1, kernel_initializer=kernel_init)(emb)
    emb = layers.Reshape((4, 4, 1))(emb)  # shape (4,4,1) as conditioning map

    # conv stack on image
    x = layers.Conv2D(cfg.disc_filters, 4, strides=2, padding="same", kernel_initializer=kernel_init)(img_in)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(cfg.disc_filters * 2, 4, strides=2, padding="same", kernel_initializer=kernel_init)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(cfg.disc_filters * 4, 4, strides=2, padding="same", kernel_initializer=kernel_init)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.25)(x)

    # Now incorporate embedding map: project x to 4x4 spatial (if needed) and concat
    x_small = layers.GlobalAveragePooling2D()(x)  # currently unused in concat strategy below, but keep
    # instead, project embeddings to match flattened conv features, or tile and concat
    # We will flatten conv features and concat flattened embedding
    flat = layers.Flatten()(x)
    emb_flat = layers.Flatten()(emb)
    combined = layers.Concatenate()([flat, emb_flat])
    out = layers.Dense(1, activation=None, kernel_initializer=kernel_init)(combined)  # linear score

    return Model([img_in, label_in], out, name="discriminator_conditional")


# ---------------- HINGE LOSS ----------------
def d_hinge_loss(real_scores: tf.Tensor, fake_scores: tf.Tensor) -> tf.Tensor:
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_scores))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_scores))
    return real_loss + fake_loss

def g_hinge_loss(fake_scores: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_scores)

# ---------------- CONDITIONAL GAN CLASS ----------------
class ConditionalGAN:
    def __init__(
        self,
        config: CGANConfig,
        generator_fn: Optional[Callable[[CGANConfig], Model]] = None,
        discriminator_fn: Optional[Callable[[CGANConfig], Model]] = None,
    ):
        self.cfg = config
        tf.random.set_seed(config.seed)
        np.random.seed(config.seed)

        self.generator = generator_fn(config) if generator_fn else build_cifar_generator(config)
        self.discriminator = discriminator_fn(config) if discriminator_fn else build_cifar_discriminator(config)

        # optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(config.lr, beta_1=config.beta_1)
        self.d_optimizer = tf.keras.optimizers.Adam(config.lr, beta_1=config.beta_1)

        # history
        self.history: Dict[str, List[float]] = {
            "d_loss": [], "g_loss": [],
            "d_real_acc": [], "d_fake_acc": [],
            # overall metrics
            "d_precision": [], "d_recall": [], "d_f1": []
        }

        # per-class history (dict of lists)
        self.per_class: Dict[int, Dict[str, List[float]]] = {
            c: {"precision": [], "recall": [], "f1": [], "real_acc": [], "fake_acc": []}
            for c in range(self.cfg.classes)
        }

        os.makedirs(self.cfg.image_folder, exist_ok=True)

    @tf.function
    def discriminator_step(self, real_images: tf.Tensor, real_labels: tf.Tensor, flip_label: bool):
        batch_size = tf.shape(real_images)[0]
        z = tf.random.normal((batch_size, self.cfg.latent_dim))
        # sample labels for fake images by reusing real labels or sampling uniformly (we'll condition generator with labels)
        fake_labels = real_labels  # simpler: try to generate same-class fakes within batch
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, fake_labels], training=True)  # generate conditioned on labels
            real_scores = self.discriminator([real_images, real_labels], training=True)  # shape (B,1)
            fake_scores = self.discriminator([fake_images, fake_labels], training=True)

            if flip_label:
                d_loss = d_hinge_loss(fake_scores, real_scores)
            else:
                d_loss = d_hinge_loss(real_scores, fake_scores)

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # compute accuracies using threshold 0 (hinge uses 0 as decision boundary)
        real_acc = tf.reduce_mean(tf.cast(real_scores > 0.0, tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(fake_scores < 0.0, tf.float32))
        return d_loss, real_scores, fake_scores, real_acc, fake_acc

    @tf.function
    def generator_step(self, labels: tf.Tensor):
        batch_size = tf.shape(labels)[0]
        z = tf.random.normal((batch_size, self.cfg.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)
            g_loss = g_hinge_loss(fake_scores)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return g_loss

    def generate_conditional(self, class_id: int, n: int):
        z = tf.random.normal((n, self.cfg.latent_dim))
        labels = np.full((n,), class_id, dtype=np.int32)
        imgs = self.generator([z, tf.constant(labels)], training=False)
        imgs = (imgs.numpy() + 1.0) / 2.0
        return imgs

    def save_grid(self, images: np.ndarray, path: str, grid=(4,4), dpi=150):
        rows, cols = grid
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
        for i, ax in enumerate(axes.flatten()):
            if i < len(images):
                ax.imshow(images[i])
            ax.axis("off")
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def evaluate_per_class(self, dataset_by_class: Dict[int, np.ndarray], n_per_class: int = 200):
        """
        For each class:
        - sample n_per_class real examples from dataset_by_class[class]
        - generate n_per_class fake images conditioned on class
        - compute precision/recall/f1 where real=1, fake=0 using D scores thresholded at 0
        - compute real_acc (fraction real_scores>0), fake_acc (fraction fake_scores<0)
        """
        precisions, recalls, f1s = [], [], []
        for c in range(self.cfg.classes):
            real_images = dataset_by_class[c]
            if len(real_images) == 0:
                # skip classes with no data (shouldn't happen in CIFAR)
                self.per_class[c]["precision"].append(0.0)
                self.per_class[c]["recall"].append(0.0)
                self.per_class[c]["f1"].append(0.0)
                self.per_class[c]["real_acc"].append(0.0)
                self.per_class[c]["fake_acc"].append(0.0)
                continue

            # sample real examples
            idx = np.random.choice(len(real_images), size=min(n_per_class, len(real_images)), replace=True)
            real_batch = real_images[idx].astype("float32")
            real_batch = (real_batch / 127.5) - 1.0  # scale to [-1,1]
            real_labels = np.full((len(real_batch),), c, dtype=np.int32)

            # generate fake images
            fake_imgs = self.generate_conditional(c, n_per_class)
            fake_labels = np.full((len(fake_imgs),), c, dtype=np.int32)

            # get discriminator scores
            real_scores = self.discriminator([real_batch, real_labels], training=False).numpy().reshape(-1)
            fake_scores = self.discriminator([fake_imgs, fake_labels], training=False).numpy().reshape(-1)

            # metrics
            y_true = np.concatenate([np.ones_like(real_scores), np.zeros_like(fake_scores)])
            y_pred = np.concatenate([(real_scores > 0.0).astype(int), (fake_scores > 0.0).astype(int)])  # >0 predicted real

            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f = f1_score(y_true, y_pred, zero_division=0)

            real_acc = np.mean((real_scores > 0.0).astype(float))
            fake_acc = np.mean((fake_scores < 0.0).astype(float))

            self.per_class[c]["precision"].append(float(p))
            self.per_class[c]["recall"].append(float(r))
            self.per_class[c]["f1"].append(float(f))
            self.per_class[c]["real_acc"].append(float(real_acc))
            self.per_class[c]["fake_acc"].append(float(fake_acc))

            precisions.append(p); recalls.append(r); f1s.append(f)

        # return aggregated averages across classes
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

    def train(self, dataset: Any, dataset_by_class: Dict[int, np.ndarray], epochs: int = 100):
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        for epoch in range(epochs):
            numeric_epoch = epoch + 1
            ep_d_losses, ep_g_losses = [], []
            ep_real_accs, ep_fake_accs = [], []
            ep_precisions, ep_recalls, ep_f1s = [], [], []

            for real_batch, real_labels in dataset:
                flip = np.random.rand() < self.cfg.flip_label_prob

                # discriminator updates (n_critic)
                for _ in range(self.cfg.n_critic):
                    d_loss, real_scores, fake_scores, real_acc, fake_acc = self.discriminator_step(real_batch, real_labels, flip)
                    ep_d_losses.append(float(d_loss.numpy()))
                    ep_real_accs.append(float(real_acc.numpy()))
                    ep_fake_accs.append(float(fake_acc.numpy()))

                # generator update (1 step conditioned on labels of batch)
                g_loss = self.generator_step(real_labels)
                ep_g_losses.append(float(g_loss.numpy()))

                # compute batch-level classification metrics (real vs fake)
                r = real_scores.numpy().reshape(-1)
                f = fake_scores.numpy().reshape(-1)
                y_true = np.concatenate([np.ones_like(r), np.zeros_like(f)])
                y_pred = np.concatenate([(r > 0.0).astype(int), (f > 0.0).astype(int)])
                ep_precisions.append(precision_score(y_true, y_pred, zero_division=0))
                ep_recalls.append(recall_score(y_true, y_pred, zero_division=0))
                ep_f1s.append(f1_score(y_true, y_pred, zero_division=0))

            # aggregate epoch stats
            d_loss_epoch = float(np.mean(ep_d_losses)) if ep_d_losses else 0.0
            g_loss_epoch = float(np.mean(ep_g_losses)) if ep_g_losses else 0.0
            real_acc_epoch = float(np.mean(ep_real_accs)) if ep_real_accs else 0.0
            fake_acc_epoch = float(np.mean(ep_fake_accs)) if ep_fake_accs else 0.0
            prec_epoch = float(np.mean(ep_precisions)) if ep_precisions else 0.0
            rec_epoch = float(np.mean(ep_recalls)) if ep_recalls else 0.0
            f1_epoch = float(np.mean(ep_f1s)) if ep_f1s else 0.0

            self.history["d_loss"].append(d_loss_epoch)
            self.history["g_loss"].append(g_loss_epoch)
            self.history["d_real_acc"].append(real_acc_epoch)
            self.history["d_fake_acc"].append(fake_acc_epoch)
            self.history["d_precision"].append(prec_epoch)
            self.history["d_recall"].append(rec_epoch)
            self.history["d_f1"].append(f1_epoch)

            # evaluate per-class on small sample set and store per_class metrics
            avg_p, avg_r, avg_f1 = self.evaluate_per_class(dataset_by_class, n_per_class=self.cfg.eval_samples_per_class)

            print(
                f"Epoch {numeric_epoch}/{epochs} | D_loss: {d_loss_epoch:.4f} | G_loss: {g_loss_epoch:.4f} | "
                f"D_real_acc: {real_acc_epoch:.3f} | D_fake_acc: {fake_acc_epoch:.3f} | "
                f"Prec: {prec_epoch:.3f} | Rec: {rec_epoch:.3f} | F1: {f1_epoch:.3f} | "
                f"PerClassAvgP: {avg_p:.3f} PerClassAvgR: {avg_r:.3f} PerClassAvgF1: {avg_f1:.3f}"
            )

            # plot progress
            ax[0].clear(); ax[0].plot(self.history["d_loss"], label="D Loss"); ax[0].plot(self.history["g_loss"], label="G Loss"); ax[0].legend(); ax[0].set_title("Loss")
            ax[1].clear(); ax[1].plot(self.history["d_real_acc"], label="D Real Acc"); ax[1].plot(self.history["d_fake_acc"], label="D Fake Acc"); ax[1].plot(self.history["d_f1"], label="D F1"); ax[1].legend(); ax[1].set_title("Discriminator Metrics")
            plt.pause(0.001)

            # periodically save sample grids per class
            if numeric_epoch % self.cfg.save_every == 0 or numeric_epoch == 1:
                for c in range(self.cfg.classes):
                    samples = self.generate_conditional(c, 16)
                    out_path = os.path.join(self.cfg.image_folder, f"epoch_{numeric_epoch}_class_{c}.png")
                    self.save_grid(samples, out_path, grid=(4,4))

        # save final history plot
        plt.ioff()
        fig2, ax2 = plt.subplots(1,2, figsize=(12,5))
        ax2[0].plot(self.history["d_loss"], label="D Loss"); ax2[0].plot(self.history["g_loss"], label="G Loss"); ax2[0].legend(); ax2[0].set_title("Loss")
        ax2[1].plot(self.history["d_real_acc"], label="D Real Acc"); ax2[1].plot(self.history["d_fake_acc"], label="D Fake Acc"); ax2[1].plot(self.history["d_f1"], label="D F1"); ax2[1].legend(); ax2[1].set_title("Discriminator Metrics")
        fig2.savefig(self.cfg.history_path, bbox_inches="tight")
        plt.close(fig2)


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load CIFAR-10 (all classes) and build dataset_by_class
    (X, y), (_, _) = tf.keras.datasets.cifar10.load_data()
    X = X.astype("float32")
    classes = 10

    # build per-class list of images (raw pixel [0,255] arrays to be scaled later in evaluation)
    dataset_by_class = {c: X[y.flatten() == c] for c in range(classes)}

    cfg = CGANConfig()
    cfg.batch_size = 64
    cfg.save_every = 50
    cfg.n_critic = 1  # hinge/DCGAN often uses 1; experiment if needed
    cfg.eval_samples_per_class = 200
    cfg.classes = classes
    cfg.image_folder = "cgan_samples"

    # Prepare tf.data dataset: yield (image_scaled, label)
    X_scaled = (X / 127.5) - 1.0  # scale to [-1,1]
    y_flat = y.flatten().astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((X_scaled, y_flat)).shuffle(10000).batch(cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    cgan = ConditionalGAN(cfg)
    print("Generator summary:")
    cgan.generator.summary()
    print("Discriminator summary:")
    cgan.discriminator.summary()

    # Train
    cgan.train(ds, dataset_by_class, epochs=500)

    # Save final sample grids for each class
    for c in range(cfg.classes):
        samples = cgan.generate_conditional(c, 64)
        out = os.path.join(cfg.image_folder, f"final_class_{c}.png")
        cgan.save_grid(samples, out, grid=(8,8))

    print("Training complete. Saved per-class sample grids and history plot.")