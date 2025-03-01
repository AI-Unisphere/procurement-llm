"""Tests for the evaluation metrics module."""

import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from procurement_granite.evaluation.metrics import ProcurementEvaluator, evaluate_model_on_test_data
from procurement_granite.utils.config import get_project_root


class TestMetrics(unittest.TestCase):
    """Tests for the evaluation metrics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the config
        self.mock_config = {
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "weights": {
                    "rfp_requirements": 0.6,
                    "best_practices": 0.4,
                },
            },
        }
        
        # Patch the config loading
        self.config_patch = patch("procurement_granite.evaluation.metrics.load_config", return_value=self.mock_config)
        self.config_patch.start()
        
        # Create the evaluator
        self.evaluator = ProcurementEvaluator()
        
        # Create test directory
        self.test_dir = Path(get_project_root()) / "tests" / "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patch.stop()
        
        # Clean up test files
        for file_path in self.test_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    
    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.evaluator, ProcurementEvaluator)
        self.assertEqual(self.evaluator.metrics, ["accuracy", "precision", "recall", "f1"])
        self.assertEqual(self.evaluator.weights, {"rfp_requirements": 0.6, "best_practices": 0.4})
    
    def test_evaluate_extraction(self):
        """Test evaluating extraction tasks."""
        predictions = ["Requirement 1", "Requirement 2"]
        ground_truth = ["Requirement 1", "Requirement 2"]
        
        metrics = self.evaluator.evaluate_extraction(predictions, ground_truth, "extract_requirements")
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Check that the metrics are reasonable
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreaterEqual(metrics["precision"], 0)
        self.assertLessEqual(metrics["precision"], 1)
        self.assertGreaterEqual(metrics["recall"], 0)
        self.assertLessEqual(metrics["recall"], 1)
        self.assertGreaterEqual(metrics["f1"], 0)
        self.assertLessEqual(metrics["f1"], 1)
    
    def test_evaluate_bid_scoring(self):
        """Test evaluating bid scoring tasks."""
        predictions = [{"score": 0.8}, {"score": 0.7}]
        ground_truth = [{"score": 0.85}, {"score": 0.75}]
        
        metrics = self.evaluator.evaluate_bid_scoring(predictions, ground_truth)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("weighted_score", metrics)
        
        # Check that the metrics are reasonable
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreaterEqual(metrics["precision"], 0)
        self.assertLessEqual(metrics["precision"], 1)
        self.assertGreaterEqual(metrics["recall"], 0)
        self.assertLessEqual(metrics["recall"], 1)
        self.assertGreaterEqual(metrics["f1"], 0)
        self.assertLessEqual(metrics["f1"], 1)
        self.assertGreaterEqual(metrics["weighted_score"], 0)
        self.assertLessEqual(metrics["weighted_score"], 1)
    
    def test_calculate_weighted_score(self):
        """Test calculating a weighted score."""
        metrics = {
            "accuracy": 0.8,
            "precision": 0.85,
            "recall": 0.75,
            "f1": 0.8,
        }
        
        weighted_score = self.evaluator.calculate_weighted_score(metrics)
        
        # Check that the weighted score is reasonable
        self.assertGreaterEqual(weighted_score, 0)
        self.assertLessEqual(weighted_score, 1)
        
        # Check that the weighted score is calculated correctly
        expected_score = 0.6 * metrics["accuracy"] + 0.4 * metrics["f1"]
        self.assertAlmostEqual(weighted_score, expected_score)
    
    def test_evaluate_model(self):
        """Test evaluating the model on multiple tasks."""
        model_outputs = {
            "extract_requirements": ["Requirement 1", "Requirement 2"],
            "identify_evaluation_criteria": ["Criterion 1", "Criterion 2"],
            "extract_technical_specifications": ["Spec 1", "Spec 2"],
            "score_bid": [{"score": 0.8}, {"score": 0.7}],
        }
        
        ground_truth = {
            "extract_requirements": ["Requirement 1", "Requirement 2"],
            "identify_evaluation_criteria": ["Criterion 1", "Criterion 2"],
            "extract_technical_specifications": ["Spec 1", "Spec 2"],
            "score_bid": [{"score": 0.85}, {"score": 0.75}],
        }
        
        results = self.evaluator.evaluate_model(model_outputs, ground_truth)
        
        # Check that the results include all tasks and overall metrics
        self.assertIn("extract_requirements", results)
        self.assertIn("identify_evaluation_criteria", results)
        self.assertIn("extract_technical_specifications", results)
        self.assertIn("score_bid", results)
        self.assertIn("overall", results)
        
        # Check that each task has the expected metrics
        for task in ["extract_requirements", "identify_evaluation_criteria", "extract_technical_specifications"]:
            self.assertIn("accuracy", results[task])
            self.assertIn("precision", results[task])
            self.assertIn("recall", results[task])
            self.assertIn("f1", results[task])
        
        # Check that score_bid has the expected metrics
        self.assertIn("accuracy", results["score_bid"])
        self.assertIn("precision", results["score_bid"])
        self.assertIn("recall", results["score_bid"])
        self.assertIn("f1", results["score_bid"])
        self.assertIn("weighted_score", results["score_bid"])
        
        # Check that overall has the expected metrics
        self.assertIn("accuracy", results["overall"])
        self.assertIn("precision", results["overall"])
        self.assertIn("recall", results["overall"])
        self.assertIn("f1", results["overall"])
        self.assertIn("weighted_score", results["overall"])
    
    @patch("json.dump")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_evaluation_results(self, mock_file, mock_json_dump):
        """Test saving evaluation results."""
        results = {
            "task1": {"metric1": 0.8, "metric2": 0.9},
            "task2": {"metric1": 0.7, "metric2": 0.8},
            "overall": {"metric1": 0.75, "metric2": 0.85},
        }
        
        output_path = os.path.join(self.test_dir, "evaluation_results.json")
        
        self.evaluator.save_evaluation_results(results, output_path)
        
        # Check that the file was opened
        mock_file.assert_called_once_with(output_path, "w")
        
        # Check that json.dump was called
        mock_json_dump.assert_called_once_with(results, mock_file(), indent=2)
    
    def test_evaluate_model_on_test_data(self):
        """Test evaluating the model on test data."""
        with patch("procurement_granite.evaluation.metrics.ProcurementEvaluator") as mock_evaluator_class:
            # Mock the evaluator
            mock_evaluator = MagicMock()
            mock_evaluator_class.return_value = mock_evaluator
            
            # Mock evaluate_model
            mock_results = {
                "task1": {"metric1": 0.8, "metric2": 0.9},
                "task2": {"metric1": 0.7, "metric2": 0.8},
                "overall": {"metric1": 0.75, "metric2": 0.85},
            }
            mock_evaluator.evaluate_model.return_value = mock_results
            
            # Call the function
            results = evaluate_model_on_test_data(
                model_path="model_path",
                test_data_path="test_data_path",
                output_path="output_path",
                config_path="config_path",
            )
            
            # Check that the evaluator was created
            mock_evaluator_class.assert_called_once_with("config_path")
            
            # Check that evaluate_model was called
            mock_evaluator.evaluate_model.assert_called_once()
            
            # Check that save_evaluation_results was called
            mock_evaluator.save_evaluation_results.assert_called_once_with(mock_results, "output_path")
            
            # Check that the results were returned
            self.assertEqual(results, mock_results)


if __name__ == "__main__":
    unittest.main() 