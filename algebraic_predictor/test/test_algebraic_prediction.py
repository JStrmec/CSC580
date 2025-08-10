import unittest
import os
from algebraic_predictor.dataset_augmentor import AlgebraicDatasetGenerator


class TestAlgebraicPrediction(unittest.TestCase):
    def setUp(self):
        self.resources_dir = os.getenv("RESOURCES_DIR", "algebraic_predictor/resources")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def test_data_augmentation(self):
        # Create dataset generator
        generator = AlgebraicDatasetGenerator(
            num_variables=4, total_examples=1000, train_ratio=0.9
        )

        # Get train/test splits
        X_train, y_train, X_test, y_test = generator.get_train_test()

        self.assertEqual(len(X_train), 900)
        self.assertEqual(len(y_train), 900)
        self.assertEqual(len(X_test), 100)
        self.assertEqual(len(y_test), 100)

        # Check variable count
        self.assertEqual(len(X_train[0]), 4)
        self.assertEqual(len(X_test[0]), 4)

        # Check output values
        for i in range(100):
            self.assertIsInstance(y_train[i], int)
            self.assertIsInstance(y_test[i], int)

        file = os.path.join(self.resources_dir, "algebraic_dataset.csv")
        generator.save_to_file(file)

        self.assertTrue(os.path.exists(file), "Dataset file was not created.")

        # Check if the dataset is generated
        self.assertGreater(len(generator.dataset), 0)
