from random import randint
from sklearn.linear_model import LinearRegression
from typing import List


class LinearEquationModel:
    def __init__(
        self,
        num_vars: int,
        coefficients: List[float],
        train_set_limit: int = 1000,
        train_set_count: int = 1000,
    ):
        if not (4 <= num_vars <= 8):
            raise ValueError("Number of variables must be between 4 and 8.")
        if len(coefficients) != num_vars:
            raise ValueError("Number of coefficients must match number of variables.")

        self.num_vars = num_vars
        self.coefficients = coefficients
        self.train_set_limit = train_set_limit
        self.train_set_count = train_set_count
        self.model = LinearRegression(n_jobs=-1)
        self.train_input = []
        self.train_output = []

    def generate_training_data(self) -> None:
        self.train_input.clear()
        self.train_output.clear()

        for _ in range(self.train_set_count):
            inputs = [randint(0, self.train_set_limit) for _ in range(self.num_vars)]
            output = sum(coef * val for coef, val in zip(self.coefficients, inputs))
            self.train_input.append(inputs)
            self.train_output.append(output)

    def train_model(self) -> None:
        if not self.train_input or not self.train_output:
            raise ValueError("Training data has not been generated yet.")
        self.model.fit(X=self.train_input, y=self.train_output)

    def predict(self, inputs: List[float]) -> float:
        if len(inputs) != self.num_vars:
            raise ValueError("Number of inputs must match number of variables.")
        return self.model.predict([inputs])[0]

    def actual_value(self, inputs: List[float]) -> float:
        return sum(coef * val for coef, val in zip(self.coefficients, inputs))

    def get_learned_coefficients(self) -> List[float]:
        """Return the coefficients learned by the model."""
        return self.model.coef_


# Example usage
if __name__ == "__main__":
    # Step 1: User input
    num_vars = int(input("Enter number of variables (4–8): "))
    coefficients = []
    for i in range(num_vars):
        coef = float(input(f"Enter coefficient for x{i + 1}: "))
        coefficients.append(coef)

    # Step 2: Initialize and train model
    model = LinearEquationModel(num_vars=num_vars, coefficients=coefficients)
    model.generate_training_data()
    model.train_model()

    # Step 3: Prediction test
    test_values = []
    for i in range(num_vars):
        val = float(input(f"Enter value for x{i + 1}: "))
        test_values.append(val)

    predicted = model.predict(test_values)
    actual = model.actual_value(test_values)

    print("\n--- Results ---")
    print(f"Predicted value: {predicted}")
    print(f"Actual value:    {actual}")
    print(f"Learned coefficients: {model.get_learned_coefficients()}")
