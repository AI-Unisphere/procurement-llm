"""Tests for the trainer module."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from datasets import Dataset

from procurement_granite.model.trainer import GraniteTrainer, train_model
from procurement_granite.utils.config import get_project_root


class TestTrainer(unittest.TestCase):
    """Tests for the trainer module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the config
        self.mock_config = {
            "model": {
                "base_model": "ibm/granite-3.2-8b-instruct",
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj"],
                },
                "quantization": None,
            },
            "paths": {
                "models": "models",
            },
            "training": {
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2.0e-4,
                "warmup_steps": 100,
                "max_steps": 1000,
                "save_steps": 200,
                "eval_steps": 200,
                "logging_steps": 50,
                "max_grad_norm": 0.3,
                "weight_decay": 0.01,
                "lr_scheduler_type": "cosine",
                "seed": 42,
            },
            "data_processing": {
                "max_length": 2048,
            },
        }
        
        # Patch the config loading
        self.config_patch = patch("procurement_granite.model.trainer.load_config", return_value=self.mock_config)
        self.config_patch.start()
        
        # Patch torch.cuda.is_available
        self.cuda_patch = patch("torch.cuda.is_available", return_value=False)
        self.cuda_patch.start()
        
        # Mock the model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.eos_token = "</s>"
        self.mock_tokenizer.pad_token = "</s>"
        
        # Patch the model loading
        self.model_patch = patch("procurement_granite.model.trainer.AutoModelForCausalLM.from_pretrained", return_value=self.mock_model)
        self.tokenizer_patch = patch("procurement_granite.model.trainer.AutoTokenizer.from_pretrained", return_value=self.mock_tokenizer)
        self.peft_patch = patch("procurement_granite.model.trainer.get_peft_model", return_value=self.mock_model)
        self.prepare_patch = patch("procurement_granite.model.trainer.prepare_model_for_kbit_training", return_value=self.mock_model)
        
        self.model_patch.start()
        self.tokenizer_patch.start()
        self.peft_patch.start()
        self.prepare_patch.start()
        
        # Create the trainer
        self.trainer = GraniteTrainer()
        
        # Create test directory
        self.test_dir = Path(get_project_root()) / "tests" / "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patch.stop()
        self.cuda_patch.stop()
        self.model_patch.stop()
        self.tokenizer_patch.stop()
        self.peft_patch.stop()
        self.prepare_patch.stop()
        
        # Clean up test files
        for file_path in self.test_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    
    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.trainer, GraniteTrainer)
        self.assertEqual(self.trainer.model_name, "ibm/granite-3.2-8b-instruct")
        self.assertEqual(self.trainer.device, torch.device("cpu"))
    
    def test_load_model_and_tokenizer(self):
        """Test loading the model and tokenizer."""
        model, tokenizer = self.trainer.load_model_and_tokenizer()
        
        self.assertEqual(model, self.mock_model)
        self.assertEqual(tokenizer, self.mock_tokenizer)
        
        # Check that the model loading functions were called
        self.model_patch.assert_called_once()
        self.tokenizer_patch.assert_called_once()
        self.peft_patch.assert_called_once()
    
    @patch("procurement_granite.model.trainer.load_dataset")
    def test_prepare_dataset(self, mock_load_dataset):
        """Test preparing the dataset."""
        # Mock load_dataset
        mock_train_dataset = Dataset.from_dict({"instruction": ["Instruction 1"], "input": ["Input 1"], "output": ["Output 1"]})
        mock_val_dataset = Dataset.from_dict({"instruction": ["Instruction 2"], "input": ["Input 2"], "output": ["Output 2"]})
        mock_synthetic_dataset = Dataset.from_dict({"instruction": ["Instruction 3"], "input": ["Input 3"], "output": ["Output 3"]})
        
        mock_load_dataset.side_effect = [mock_train_dataset, mock_val_dataset, mock_synthetic_dataset]
        
        # Load model and tokenizer
        self.trainer.load_model_and_tokenizer()
        
        # Mock tokenizer functions
        self.mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        
        # Prepare dataset
        with patch("os.path.exists", return_value=True):
            train_dataset, val_dataset = self.trainer.prepare_dataset(
                "train.json",
                "val.json",
                "synthetic.json",
            )
        
        # Check that load_dataset was called
        self.assertEqual(mock_load_dataset.call_count, 3)
        
        # Check that the datasets were processed
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
    
    @patch("procurement_granite.model.trainer.Trainer")
    @patch("procurement_granite.model.trainer.TrainingArguments")
    @patch("procurement_granite.model.trainer.DataCollatorForSeq2Seq")
    @patch("procurement_granite.model.trainer.get_last_checkpoint")
    def test_train(self, mock_get_last_checkpoint, mock_data_collator, mock_training_args, mock_trainer):
        """Test training the model."""
        # Mock get_last_checkpoint
        mock_get_last_checkpoint.return_value = None
        
        # Mock DataCollatorForSeq2Seq
        mock_collator = MagicMock()
        mock_data_collator.return_value = mock_collator
        
        # Mock TrainingArguments
        mock_args = MagicMock()
        mock_training_args.return_value = mock_args
        
        # Mock Trainer
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Load model and tokenizer
        self.trainer.load_model_and_tokenizer()
        
        # Create mock datasets
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        
        # Train model
        model = self.trainer.train(mock_train_dataset, mock_val_dataset)
        
        # Check that the trainer was created and called
        mock_trainer.assert_called_once_with(
            model=self.mock_model,
            args=mock_args,
            train_dataset=mock_train_dataset,
            eval_dataset=mock_val_dataset,
            data_collator=mock_collator,
        )
        
        mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=None)
        
        # Check that the model was saved
        self.assertEqual(self.mock_model.save_pretrained.call_count, 1)
        self.assertEqual(self.mock_tokenizer.save_pretrained.call_count, 1)
    
    @patch("procurement_granite.model.trainer.GraniteTrainer.run_training_pipeline")
    @patch("procurement_granite.model.trainer.os.path.join")
    def test_train_model(self, mock_join, mock_run_pipeline):
        """Test the train_model function."""
        # Mock os.path.join
        mock_join.return_value = "mock_path"
        
        # Mock run_training_pipeline
        mock_model = MagicMock()
        mock_run_pipeline.return_value = mock_model
        
        # Train model
        model = train_model()
        
        # Check that run_training_pipeline was called
        mock_run_pipeline.assert_called_once_with("mock_path", "mock_path", "mock_path")
        
        # Check that the model was returned
        self.assertEqual(model, mock_model)


if __name__ == "__main__":
    unittest.main() 