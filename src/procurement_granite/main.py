"""Main script for running the procurement-granite pipeline."""

import argparse
import os
from pathlib import Path
from typing import Optional

from procurement_granite.data.document_processor import process_raw_documents
from procurement_granite.data.dataset_creator import create_datasets_from_processed_documents
from procurement_granite.model.trainer import train_model
from procurement_granite.model.inference import load_inference_model
from procurement_granite.evaluation.metrics import evaluate_model_on_test_data
from procurement_granite.api.app import start_server
from procurement_granite.utils.config import get_project_root, load_config


def process_data(config_path: Optional[str] = None):
    """Process raw documents and create datasets.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.
    """
    print("Processing raw documents...")
    documents = process_raw_documents()
    
    print("Creating datasets...")
    train_df, val_df, synthetic_df = create_datasets_from_processed_documents(
        create_synthetic=True,
        num_synthetic=100,
    )
    
    print(f"Created {len(train_df)} training examples, {len(val_df)} validation examples, "
          f"and {len(synthetic_df) if synthetic_df is not None else 0} synthetic examples.")


def train(config_path: Optional[str] = None):
    """Train the model.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.
    """
    print("Training model...")
    model = train_model(config_path=config_path)
    print("Model training complete.")


def evaluate(model_path: Optional[str] = None, config_path: Optional[str] = None):
    """Evaluate the model.
    
    Args:
        model_path: Path to the model. If None, uses the default path.
        config_path: Path to the configuration file. If None, uses the default config.
    """
    print("Evaluating model...")
    results = evaluate_model_on_test_data(
        model_path=model_path,
        config_path=config_path,
    )
    
    print("Evaluation results:")
    for task, metrics in results.items():
        print(f"  {task}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")


def serve(model_path: Optional[str] = None, config_path: Optional[str] = None):
    """Start the API server.
    
    Args:
        model_path: Path to the model. If None, uses the default path.
        config_path: Path to the configuration file. If None, uses the default config.
    """
    print("Starting API server...")
    start_server()


def run_pipeline(config_path: Optional[str] = None):
    """Run the complete pipeline.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.
    """
    process_data(config_path)
    train(config_path)
    evaluate(config_path=config_path)
    serve(config_path=config_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Procurement Document Analysis with IBM GRANITE")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process data command
    process_parser = subparsers.add_parser("process", help="Process raw documents and create datasets")
    process_parser.add_argument("--config", help="Path to configuration file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", help="Path to configuration file")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument("--model", help="Path to model")
    evaluate_parser.add_argument("--config", help="Path to configuration file")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--model", help="Path to model")
    serve_parser.add_argument("--config", help="Path to configuration file")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the complete pipeline")
    pipeline_parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_data(args.config)
    elif args.command == "train":
        train(args.config)
    elif args.command == "evaluate":
        evaluate(args.model, args.config)
    elif args.command == "serve":
        serve(args.model, args.config)
    elif args.command == "pipeline":
        run_pipeline(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 