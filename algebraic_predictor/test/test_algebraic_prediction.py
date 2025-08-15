import unittest
from algebraic_predictor.algebraic_predictor import LinearEquationModel


class TestLinearEquationModel(unittest.TestCase):
    def setUp(self):
        self.num_vars = 4
        self.coefficients = [1.0, 2.0, 3.0, 4.0]
        self.model = LinearEquationModel(
            num_vars=self.num_vars,
            coefficients=self.coefficients,
            train_set_limit=10,
            train_set_count=200,
        )
        self.model.generate_training_data()
        self.model.train_model()

    def test_initialization(self):
        self.assertEqual(self.model.num_vars, self.num_vars)
        self.assertEqual(self.model.coefficients, self.coefficients)

    def test_generate_training_data_length(self):
        self.assertEqual(len(self.model.train_input), 200)
        self.assertEqual(len(self.model.train_output), 200)

    def test_training_and_prediction(self):
        test_input = [1, 2, 3, 4]
        predicted = self.model.predict(test_input)
        actual = self.model.actual_value(test_input)
        self.assertAlmostEqual(predicted, actual, places=5)

    def test_actual_value_computation(self):
        test_input = [1, 1, 1, 1]
        expected = sum(self.coefficients)
        actual = self.model.actual_value(test_input)
        self.assertEqual(actual, expected)

    def test_learned_coefficients(self):
        learned = self.model.get_learned_coefficients()
        for learned_c, actual_c in zip(learned, self.coefficients):
            self.assertAlmostEqual(learned_c, actual_c, places=5)

    def test_invalid_initialization(self):
        with self.assertRaises(ValueError):
            LinearEquationModel(num_vars=3, coefficients=[1, 2, 3])

    def test_invalid_prediction_input_length(self):
        with self.assertRaises(ValueError):
            self.model.predict([1, 2])  # too short
