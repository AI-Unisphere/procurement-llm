"""Evaluation metrics for the procurement-granite project."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from procurement_granite.utils.config import get_project_root, load_config


class ProcurementEvaluator:
    """Evaluator for procurement document analysis tasks."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the evaluator.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default config.
        """
        self.config = load_config(config_path)
        self.metrics = self.config["evaluation"]["metrics"]
        self.weights = self.config["evaluation"]["weights"]
    
    def evaluate_extraction(
        self, 
        predictions: List[str], 
        ground_truth: List[str],
        task: str = "extract_requirements",
    ) -> Dict[str, float]:
        """Evaluate extraction tasks.
        
        Args:
            predictions: List of predicted extractions.
            ground_truth: List of ground truth extractions.
            task: The extraction task.
            
        Returns:
            Dict[str, float]: The evaluation metrics.
        """
        # This is a placeholder for more sophisticated evaluation
        # In a real implementation, this would use techniques like RAGAS or custom metrics
        
        # For now, we'll just return placeholder metrics
        return {
            "accuracy": 0.85,
            "precision": 0.87,
            "recall": 0.83,
            "f1": 0.85,
        }
    
    def evaluate_bid_scoring(
        self, 
        predictions: List[Dict], 
        ground_truth: List[Dict],
    ) -> Dict[str, float]:
        """Evaluate bid scoring tasks.
        
        Args:
            predictions: List of predicted bid scores.
            ground_truth: List of ground truth bid scores.
            
        Returns:
            Dict[str, float]: The evaluation metrics.
        """
        # This is a placeholder for more sophisticated evaluation
        # In a real implementation, this would compare the predicted scores with expert scores
        
        # For now, we'll just return placeholder metrics
        return {
            "accuracy": 0.80,
            "precision": 0.82,
            "recall": 0.78,
            "f1": 0.80,
            "weighted_score": 0.81,
        }
    
    def calculate_weighted_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a weighted score from multiple metrics.
        
        Args:
            metrics: Dictionary of metrics.
            
        Returns:
            float: The weighted score.
        """
        # Apply weights to RFP requirements and best practices
        rfp_weight = self.weights["rfp_requirements"]
        bp_weight = self.weights["best_practices"]
        
        # This is a placeholder for a more sophisticated weighted scoring
        # In a real implementation, this would use the actual metrics and weights
        
        weighted_score = (
            rfp_weight * metrics.get("accuracy", 0) + 
            bp_weight * metrics.get("f1", 0)
        )
        
        return weighted_score
    
    def evaluate_model(
        self, 
        model_outputs: Dict[str, List], 
        ground_truth: Dict[str, List],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the model on multiple tasks.
        
        Args:
            model_outputs: Dictionary mapping tasks to lists of model outputs.
            ground_truth: Dictionary mapping tasks to lists of ground truth.
            
        Returns:
            Dict[str, Dict[str, float]]: The evaluation metrics for each task.
        """
        results = {}
        
        # Evaluate extraction tasks
        for task in ["extract_requirements", "identify_evaluation_criteria", "extract_technical_specifications"]:
            if task in model_outputs and task in ground_truth:
                results[task] = self.evaluate_extraction(
                    model_outputs[task],
                    ground_truth[task],
                    task,
                )
        
        # Evaluate bid scoring tasks
        if "score_bid" in model_outputs and "score_bid" in ground_truth:
            results["score_bid"] = self.evaluate_bid_scoring(
                model_outputs["score_bid"],
                ground_truth["score_bid"],
            )
        
        # Calculate overall score
        if results:
            overall_metrics = {
                metric: np.mean([task_results.get(metric, 0) for task_results in results.values()])
                for metric in self.metrics
            }
            
            overall_metrics["weighted_score"] = self.calculate_weighted_score(overall_metrics)
            results["overall"] = overall_metrics
        
        return results
    
    def save_evaluation_results(
        self, 
        results: Dict[str, Dict[str, float]], 
        output_path: Optional[Union[str, Path]] = None,
    ):
        """Save evaluation results to a file.
        
        Args:
            results: The evaluation results.
            output_path: Path to save the results. If None, uses the default path.
        """
        if output_path is None:
            output_path = os.path.join(
                get_project_root(),
                "evaluation_results.json",
            )
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")


def evaluate_model_on_test_data(
    model_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the model on test data.
    
    Args:
        model_path: Path to the model. If None, uses the default path.
        test_data_path: Path to the test data. If None, uses the default path.
        output_path: Path to save the evaluation results. If None, uses the default path.
        config_path: Path to the configuration file. If None, uses the default config.
        
    Returns:
        Dict[str, Dict[str, float]]: The evaluation results.
    """
    # This is a placeholder for the actual evaluation
    # In a real implementation, this would load the model and test data,
    # run the model on the test data, and evaluate the results
    
    # Create a dummy evaluator
    evaluator = ProcurementEvaluator(config_path)
    
    # Create dummy model outputs and ground truth
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
    
    # Evaluate the model
    results = evaluator.evaluate_model(model_outputs, ground_truth)
    
    # Save the results
    if output_path is not None:
        evaluator.save_evaluation_results(results, output_path)
    
    return results 