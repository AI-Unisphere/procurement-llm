"""Dataset creation utilities for the procurement-granite project."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from procurement_granite.utils.config import get_data_path, load_config


class DatasetCreator:
    """Create instruction-tuning datasets for fine-tuning the model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the dataset creator.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default config.
        """
        self.config = load_config(config_path)
        self.max_length = self.config["data_processing"]["max_length"]
        self.train_test_split_ratio = self.config["data_processing"]["train_test_split"]
    
    def create_instruction_dataset(
        self, 
        documents: Dict[str, str], 
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create an instruction-tuning dataset from processed documents.
        
        Args:
            documents: Dictionary mapping document names to their text content.
            output_dir: Directory to save the dataset files. If None, uses the default processed data directory.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The training and validation datasets.
        """
        if output_dir is None:
            output_dir = get_data_path("processed")
        
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create instruction-tuning examples
        examples = []
        
        for doc_name, doc_text in documents.items():
            # Truncate document if it's too long
            if len(doc_text) > self.max_length:
                doc_text = doc_text[:self.max_length]
            
            # Create examples for different procurement tasks
            examples.extend(self._create_rfp_analysis_examples(doc_name, doc_text))
            examples.extend(self._create_bid_evaluation_examples(doc_name, doc_text))
        
        # Convert to DataFrame
        df = pd.DataFrame(examples)
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            df, 
            train_size=self.train_test_split_ratio, 
            random_state=self.config["training"]["seed"]
        )
        
        # Save datasets
        train_df.to_json(output_dir / "train.json", orient="records", lines=True)
        val_df.to_json(output_dir / "validation.json", orient="records", lines=True)
        
        return train_df, val_df
    
    def _create_rfp_analysis_examples(self, doc_name: str, doc_text: str) -> List[Dict]:
        """Create examples for RFP analysis tasks.
        
        Args:
            doc_name: Name of the document.
            doc_text: Text content of the document.
            
        Returns:
            List[Dict]: List of examples for RFP analysis.
        """
        examples = []
        
        # Example 1: Extract key requirements
        examples.append({
            "instruction": "Extract the key requirements from this RFP document.",
            "input": doc_text,
            "output": "This is a placeholder for extracted requirements. In a real scenario, this would be annotated by procurement experts.",
            "doc_name": doc_name,
            "task": "extract_requirements"
        })
        
        # Example 2: Identify evaluation criteria
        examples.append({
            "instruction": "Identify the evaluation criteria and their weights from this RFP document.",
            "input": doc_text,
            "output": "This is a placeholder for evaluation criteria. In a real scenario, this would be annotated by procurement experts.",
            "doc_name": doc_name,
            "task": "identify_evaluation_criteria"
        })
        
        # Example 3: Extract technical specifications
        examples.append({
            "instruction": "Extract the technical specifications for the required connectivity services from this RFP document.",
            "input": doc_text,
            "output": "This is a placeholder for technical specifications. In a real scenario, this would be annotated by procurement experts.",
            "doc_name": doc_name,
            "task": "extract_technical_specifications"
        })
        
        return examples
    
    def _create_bid_evaluation_examples(self, doc_name: str, doc_text: str) -> List[Dict]:
        """Create examples for bid evaluation tasks.
        
        Args:
            doc_name: Name of the document.
            doc_text: Text content of the document.
            
        Returns:
            List[Dict]: List of examples for bid evaluation.
        """
        examples = []
        
        # Example 1: Score bid against requirements
        examples.append({
            "instruction": "Score this bid against the RFP requirements. The RFP is: {rfp_text}. The bid is: {bid_text}",
            "input": f"rfp_text: {doc_text}\nbid_text: This is a placeholder for a bid document.",
            "output": "This is a placeholder for bid scoring. In a real scenario, this would be annotated by procurement experts.",
            "doc_name": doc_name,
            "task": "score_bid"
        })
        
        # Example 2: Identify strengths and weaknesses
        examples.append({
            "instruction": "Identify the strengths and weaknesses of this bid proposal against the RFP requirements. The RFP is: {rfp_text}. The bid is: {bid_text}",
            "input": f"rfp_text: {doc_text}\nbid_text: This is a placeholder for a bid document.",
            "output": "This is a placeholder for strengths and weaknesses analysis. In a real scenario, this would be annotated by procurement experts.",
            "doc_name": doc_name,
            "task": "identify_strengths_weaknesses"
        })
        
        return examples
    
    def create_synthetic_examples(
        self, 
        num_examples: int = 100, 
        output_dir: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Create synthetic examples to enhance the training data.
        
        Args:
            num_examples: Number of synthetic examples to create.
            output_dir: Directory to save the synthetic examples. If None, uses the default synthetic data directory.
            
        Returns:
            pd.DataFrame: The synthetic examples dataset.
        """
        # This is a placeholder for synthetic data generation
        # In a real implementation, this would use techniques like templates, rules, or even another LLM
        # to generate synthetic examples
        
        if output_dir is None:
            output_dir = get_data_path("synthetic")
        
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Placeholder for synthetic examples
        examples = []
        for i in range(num_examples):
            examples.append({
                "instruction": f"Synthetic instruction {i}",
                "input": f"Synthetic input {i}",
                "output": f"Synthetic output {i}",
                "doc_name": f"synthetic_{i}",
                "task": "synthetic"
            })
        
        df = pd.DataFrame(examples)
        df.to_json(output_dir / "synthetic_examples.json", orient="records", lines=True)
        
        return df


def create_datasets_from_processed_documents(
    output_dir: Optional[Union[str, Path]] = None,
    create_synthetic: bool = True,
    num_synthetic: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Create datasets from processed documents.
    
    Args:
        output_dir: Directory to save the dataset files. If None, uses the default processed data directory.
        create_synthetic: Whether to create synthetic examples.
        num_synthetic: Number of synthetic examples to create.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: The training, validation, and synthetic datasets.
    """
    if output_dir is None:
        output_dir = get_data_path("processed")
    
    # Load processed documents
    processed_dir = get_data_path("processed")
    documents = {}
    
    for file_path in processed_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents[file_path.name] = f.read()
    
    # Create instruction datasets
    creator = DatasetCreator()
    train_df, val_df = creator.create_instruction_dataset(documents, output_dir)
    
    # Create synthetic examples if requested
    synthetic_df = None
    if create_synthetic:
        synthetic_df = creator.create_synthetic_examples(num_synthetic)
    
    return train_df, val_df, synthetic_df 