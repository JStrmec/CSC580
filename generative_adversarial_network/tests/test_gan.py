import os
import tempfile
import unittest
import numpy as np
import tensorflow as tf
from gan.gan import GAN, GANConfig
import warnings

warnings.filterwarnings("ignore", message=".*retracing.*")


class TestGAN(unittest.TestCase):
    def setUp(self):
        # Use a reproducible RNG
        self.rng = np.random.default_rng(seed=42)

        cfg = GANConfig(latent_dim=8, image_shape=(8, 8, 1))
        self.gan = GAN(config=cfg)
        self.gan.compile()

    def test_instantiation_and_shapes(self):
        z = self.rng.uniform(-1, 1, (2, self.gan.latent_dim)).astype(np.float32)
        gen_out = self.gan.generator.predict(z, verbose=0)
        self.assertEqual(gen_out.shape, (2, *self.gan.image_shape))

        disc_out = self.gan.discriminator.predict(gen_out, verbose=0)
        self.assertEqual(disc_out.shape, (2, 1))

    def test_train_step_updates_weights(self):
        batch_size = 4
        real_images = self.rng.uniform(
            -1, 1, size=(batch_size, *self.gan.image_shape)
        ).astype(np.float32)

        g_before = [w.numpy().copy() for w in self.gan.generator.trainable_variables]
        d_before = [
            w.numpy().copy() for w in self.gan.discriminator.trainable_variables
        ]

        d_loss, g_loss = self.gan.train_step(tf.constant(real_images), batch_size)

        g_after = [w.numpy() for w in self.gan.generator.trainable_variables]
        d_after = [w.numpy() for w in self.gan.discriminator.trainable_variables]

        # At least one weight should differ
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(g_before, g_after)))
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(d_before, d_after)))

    def test_save_and_load_consistency(self):
        z = self.rng.uniform(-1, 1, (1, self.gan.latent_dim)).astype(np.float32)
        out_before = self.gan.generate(n=1, latent_vectors=z)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.gan.save(tmpdir)

            gan2 = GAN(config=self.gan.config)
            gan2.load(tmpdir)

            out_after = gan2.generate(n=1, latent_vectors=z)

        np.testing.assert_allclose(out_before, out_after, atol=1e-6)

    def test_end_to_end_training_smoke(self):
        """Runs a tiny GAN training loop on synthetic data (CI smoke test)."""
        cfg = GANConfig(latent_dim=8, image_shape=(8, 8, 1))
        gan = GAN(config=cfg)
        gan.compile()

        # tiny synthetic dataset
        data = self.rng.uniform(-1, 1, size=(32, *cfg.image_shape)).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(4)

        gan.train(dataset, epochs=2, batch_size=4, verbose=False)

        samples = gan.generate(n=2)
        self.assertEqual(samples.shape, (2, *cfg.image_shape))

        with tempfile.TemporaryDirectory() as tmpdir:
            gan.save(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "generator.keras")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "discriminator.keras")))
