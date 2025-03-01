"""Tests for the dataset creator module."""

import os
import unittest
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pandas as pd

from procurement_granite.data.dataset_creator import DatasetCreator, create_datasets_from_processed_documents
from procurement_granite.utils.config import get_project_root


class TestDatasetCreator(unittest.TestCase):
    """Tests for the dataset creator module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the config
        self.mock_config = {
            "data_processing": {
                "max_length": 2048,
                "train_test_split": 0.9,
            },
            "training": {
                "seed": 42,
            },
        }
        
        # Patch the config loading
        self.config_patch = patch("procurement_granite.data.dataset_creator.load_config", return_value=self.mock_config)
        self.config_patch.start()
        
        # Create the dataset creator
        self.creator = DatasetCreator()
        
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
        self.assertIsInstance(self.creator, DatasetCreator)
        self.assertEqual(self.creator.max_length, 2048)
        self.assertEqual(self.creator.train_test_split_ratio, 0.9)
    
    def test_create_rfp_analysis_examples(self):
        """Test creating RFP analysis examples."""
        doc_name = "test_doc.txt"
        doc_text = "This is a test RFP document."
        
        examples = self.creator._create_rfp_analysis_examples(doc_name, doc_text)
        
        self.assertEqual(len(examples), 3)
        
        # Check the first example
        self.assertEqual(examples[0]["instruction"], "Extract the key requirements from this RFP document.")
        self.assertEqual(examples[0]["input"], doc_text)
        self.assertEqual(examples[0]["doc_name"], doc_name)
        self.assertEqual(examples[0]["task"], "extract_requirements")
        
        # Check the second example
        self.assertEqual(examples[1]["instruction"], "Identify the evaluation criteria and their weights from this RFP document.")
        self.assertEqual(examples[1]["input"], doc_text)
        self.assertEqual(examples[1]["doc_name"], doc_name)
        self.assertEqual(examples[1]["task"], "identify_evaluation_criteria")
        
        # Check the third example
        self.assertEqual(examples[2]["instruction"], "Extract the technical specifications for the required connectivity services from this RFP document.")
        self.assertEqual(examples[2]["input"], doc_text)
        self.assertEqual(examples[2]["doc_name"], doc_name)
        self.assertEqual(examples[2]["task"], "extract_technical_specifications")
    
    def test_create_bid_evaluation_examples(self):
        """Test creating bid evaluation examples."""
        doc_name = "test_doc.txt"
        doc_text = "This is a test RFP document."
        
        examples = self.creator._create_bid_evaluation_examples(doc_name, doc_text)
        
        self.assertEqual(len(examples), 2)
        
        # Check the first example
        self.assertEqual(examples[0]["instruction"], "Score this bid against the RFP requirements. The RFP is: {rfp_text}. The bid is: {bid_text}")
        self.assertEqual(examples[0]["input"], f"rfp_text: {doc_text}\nbid_text: This is a placeholder for a bid document.")
        self.assertEqual(examples[0]["doc_name"], doc_name)
        self.assertEqual(examples[0]["task"], "score_bid")
        
        # Check the second example
        self.assertEqual(examples[1]["instruction"], "Identify the strengths and weaknesses of this bid proposal against the RFP requirements. The RFP is: {rfp_text}. The bid is: {bid_text}")
        self.assertEqual(examples[1]["input"], f"rfp_text: {doc_text}\nbid_text: This is a placeholder for a bid document.")
        self.assertEqual(examples[1]["doc_name"], doc_name)
        self.assertEqual(examples[1]["task"], "identify_strengths_weaknesses")
    
    @patch("pandas.DataFrame.to_json")
    @patch("procurement_granite.data.dataset_creator.train_test_split")
    def test_create_instruction_dataset(self, mock_train_test_split, mock_to_json):
        """Test creating an instruction dataset."""
        # Mock train_test_split
        mock_train_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_val_df = pd.DataFrame({"col1": [4, 5]})
        mock_train_test_split.return_value = (mock_train_df, mock_val_df)
        
        # Create test documents
        documents = {
            "doc1.txt": "This is document 1.",
            "doc2.txt": "This is document 2.",
        }
        
        # Create instruction dataset
        train_df, val_df = self.creator.create_instruction_dataset(documents, self.test_dir)
        
        # Check the result
        self.assertEqual(train_df, mock_train_df)
        self.assertEqual(val_df, mock_val_df)
        
        # Check that train_test_split was called
        mock_train_test_split.assert_called_once()
        
        # Check that to_json was called twice (once for train, once for val)
        self.assertEqual(mock_to_json.call_count, 2)
    
    @patch("pandas.DataFrame.to_json")
    def test_create_synthetic_examples(self, mock_to_json):
        """Test creating synthetic examples."""
        # Create synthetic examples
        df = self.creator.create_synthetic_examples(num_examples=10, output_dir=self.test_dir)
        
        # Check the result
        self.assertEqual(len(df), 10)
        self.assertIn("instruction", df.columns)
        self.assertIn("input", df.columns)
        self.assertIn("output", df.columns)
        self.assertIn("doc_name", df.columns)
        self.assertIn("task", df.columns)
        
        # Check that to_json was called
        mock_to_json.assert_called_once()
    
    @patch("procurement_granite.data.dataset_creator.DatasetCreator.create_instruction_dataset")
    @patch("procurement_granite.data.dataset_creator.DatasetCreator.create_synthetic_examples")
    @patch("procurement_granite.data.dataset_creator.get_data_path")
    @patch("pathlib.Path.glob")
    @patch("builtins.open", new_callable=mock_open, read_data="Test document content")
    def test_create_datasets_from_processed_documents(
        self, mock_file, mock_glob, mock_get_data_path, mock_create_synthetic, mock_create_instruction
    ):
        """Test creating datasets from processed documents."""
        # Mock get_data_path
        mock_get_data_path.return_value = Path(self.test_dir)
        
        # Mock glob
        mock_path1 = MagicMock()
        mock_path1.name = "doc1.txt"
        mock_path2 = MagicMock()
        mock_path2.name = "doc2.txt"
        mock_glob.return_value = [mock_path1, mock_path2]
        
        # Mock create_instruction_dataset
        mock_train_df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_val_df = pd.DataFrame({"col1": [4, 5]})
        mock_create_instruction.return_value = (mock_train_df, mock_val_df)
        
        # Mock create_synthetic_examples
        mock_synthetic_df = pd.DataFrame({"col1": [6, 7, 8]})
        mock_create_synthetic.return_value = mock_synthetic_df
        
        # Create datasets
        train_df, val_df, synthetic_df = create_datasets_from_processed_documents(
            output_dir=self.test_dir,
            create_synthetic=True,
            num_synthetic=10,
        )
        
        # Check the result
        self.assertEqual(train_df, mock_train_df)
        self.assertEqual(val_df, mock_val_df)
        self.assertEqual(synthetic_df, mock_synthetic_df)
        
        # Check that create_instruction_dataset was called
        mock_create_instruction.assert_called_once()
        
        # Check that create_synthetic_examples was called
        mock_create_synthetic.assert_called_once_with(10)


if __name__ == "__main__":
    unittest.main() 