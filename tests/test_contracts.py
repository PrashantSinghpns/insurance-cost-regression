"""Contract tests for insurance data loading."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Import the local training module without packaging it.
from train import load_and_clean  # Test the production input contract.


class InsuranceDataContractTests(unittest.TestCase):
    """Verify that invalid targets cannot enter the model."""

    def test_preserves_decimal_bmi_and_drops_invalid_charge(self):
        rows = pd.DataFrame(
            [
                {"age": 30, "sex": "female", "bmi": 27.95, "children": 1, "smoker": "no", "region": "north", "charges": 1234.56},  # Valid decimal values.
                {"age": 31, "sex": "male", "bmi": 31.25, "children": 0, "smoker": "yes", "region": "south", "charges": 0},  # Invalid target.
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "insurance.csv"  # Use a temporary dataset path.
            rows.to_csv(path, index=False)  # Write the controlled test rows.
            cleaned = load_and_clean(path)  # Apply the production quality checks.
        self.assertEqual(len(cleaned), 1)  # Remove the invalid charge row.
        self.assertEqual(cleaned.iloc[0]["bmi"], 27.95)  # Ensure decimals are never truncated.


if __name__ == "__main__":
    unittest.main()  # Support direct local execution.
