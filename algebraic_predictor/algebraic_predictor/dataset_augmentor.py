import random
import csv
from typing import List, Tuple


class AlgebraicDatasetGenerator:
    def __init__(
        self, num_variables: int, total_examples: int = 1000, train_ratio: float = 0.9
    ):
        if not (2 <= num_variables <= 8):
            raise ValueError("Number of variables must be between 2 and 8.")
        if total_examples <= 0:
            raise ValueError("Total examples must be positive.")
        if not (0 < train_ratio < 1):
            raise ValueError("Train ratio must be between 0 and 1.")

        self.num_variables = num_variables
        self.total_examples = total_examples
        self.train_ratio = train_ratio
        self.test_ratio = 1 - train_ratio

        # Coefficients for the algebraic equation
        self.coefficients = [random.randint(1, 5) for _ in range(num_variables)]
        self.dataset = []

    def _generate_dataset(self):
        self.dataset = []
        for _ in range(self.total_examples):
            inputs = [random.randint(0, 100) for _ in range(self.num_variables)]
            output = sum(c * x for c, x in zip(self.coefficients, inputs))
            self.dataset.append((inputs, output))

    def get_train_test(
        self,
    ) -> Tuple[List[List[int]], List[int], List[List[int]], List[int]]:
        self._generate_dataset()
        split_index = int(self.train_ratio * self.total_examples)

        train_data = self.dataset[:split_index]
        test_data = self.dataset[split_index:]

        X_train = [inp for inp, out in train_data]
        y_train = [out for inp, out in train_data]
        X_test = [inp for inp, out in test_data]
        y_test = [out for inp, out in test_data]

        return X_train, y_train, X_test, y_test

    def save_to_file(self, filename: str):
        if not self.dataset:
            self._generate_dataset()
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            header = [f"x{i + 1}" for i in range(self.num_variables)] + ["y"]
            writer.writerow(header)
            for inputs, output in self.dataset:
                writer.writerow(inputs + [output])
