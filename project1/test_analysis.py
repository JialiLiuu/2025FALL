"""
Filename: test_analysis.py
Author: jiali liu
Date: 2025-09-26
Description: Unit tests for validating the time complexity analysis functions.
Version: 1.0
"""

import unittest
from analysis_module import (
    analyze_code,
    collect_data,
    normalize_with_median_ratio,
    fit_linear_regression
)

class TestAnalysisFunctions(unittest.TestCase):

    def setUp(self):
        """ 
        Define a small range of n values for testing. 
        """
        self.n_values = [1000, 5000, 10000]

    def test_analyze_code_output_type(self):
        """
        Test that analyze_code returns a float.
        """
        for n in self.n_values:
            result = analyze_code(n)
            # Check that the result is a float
            self.assertIsInstance(result, float)
            # Check that the result is not negative
            self.assertGreaterEqual(result, 0)

    def test_collect_data_output(self):
        """
        Test that collect_data returns correct lengths and types.
        """
        experimental, theoretical = collect_data(self.n_values)
        # Ensure both lists match the length of input n_values
        self.assertEqual(len(experimental), len(self.n_values))
        self.assertEqual(len(theoretical), len(self.n_values))
        # Check that all values in experimental are floats
        for t in experimental:
            self.assertIsInstance(t, float)
        # Check that all values in theoretical are floats
        for c in theoretical:
            self.assertIsInstance(c, float)

    def test_normalize_with_median_ratio(self):
        """
        Test normalization returns correct length and scaling factor.
        """
        experimental, theoretical = collect_data(self.n_values)
        normalized, scaling = normalize_with_median_ratio(experimental, theoretical)
        # Check that normalized list has same length as theoretical
        self.assertEqual(len(normalized), len(theoretical))
        # Check that scaling factor is a float
        self.assertIsInstance(scaling, float)

    def test_fit_linear_regression(self):
        """
        Test regression returns predictions and coefficients.
        """
        experimental, theoretical = collect_data(self.n_values)
        predictions, coef, intercept = fit_linear_regression(theoretical, experimental)
        # Check that predictions list matches input length
        self.assertEqual(len(predictions), len(self.n_values))
        # Check that coefficient and intercept are floats
        self.assertIsInstance(coef, float)
        self.assertIsInstance(intercept, float)

# This block runs the tests when the script is executed directly
if __name__ == "__main__":
    unittest.main()
