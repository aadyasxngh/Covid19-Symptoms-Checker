import unittest
from ml_dl_predict import predict_with_custom_input

class TestPrediction(unittest.TestCase):
    """Tests for all trained ML and DL models."""

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests."""
        cls.symptoms = ["Fever", "Dry-Cough", "Sore-Throat"]
        cls.age = "25-59"
        cls.contact = "Yes"

    def test_logistic_regression(self):
        result = predict_with_custom_input(self.symptoms, self.age, self.contact, "Logistic Regression")
        self.assertIn(result, [0, 1])

    def test_random_forest(self):
        result = predict_with_custom_input(self.symptoms, self.age, self.contact, "Random Forest")
        self.assertIn(result, [0, 1])

    def test_deep_learning_model(self):
        result = predict_with_custom_input(self.symptoms, self.age, self.contact, "Deep Learning (COVID-19)")
        self.assertIn(result, [0, 1])
