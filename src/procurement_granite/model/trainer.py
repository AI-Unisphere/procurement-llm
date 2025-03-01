"""Model training utilities for the procurement-granite project."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import get_last_checkpoint

from procurement_granite.utils.config import get_project_root, load_config


class GraniteTrainer:
    """Trainer for fine-tuning IBM GRANITE 3.2 8B Instruct with LoRA."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the trainer.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default config.
        """
        self.config = load_config(config_path)
        self.model_name = self.config["model"]["base_model"]
        self.output_dir = os.path.join(get_project_root(), self.config["paths"]["models"])
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare quantization config if needed
        quantization = self.config["model"]["quantization"]
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        
        # Load model with quantization if specified
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config["model"]["lora"]["r"],
            lora_alpha=self.config["model"]["lora"]["alpha"],
            lora_dropout=self.config["model"]["lora"]["dropout"],
            target_modules=self.config["model"]["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, train_path: str, val_path: str, synthetic_path: Optional[str] = None):
        """Prepare the dataset for training.
        
        Args:
            train_path: Path to the training dataset.
            val_path: Path to the validation dataset.
            synthetic_path: Path to the synthetic dataset. If provided, it will be combined with the training dataset.
            
        Returns:
            tuple: The prepared training and validation datasets.
        """
        # Load datasets
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        val_dataset = load_dataset("json", data_files=val_path, split="train")
        
        # Combine with synthetic data if provided
        if synthetic_path and os.path.exists(synthetic_path):
            synthetic_dataset = load_dataset("json", data_files=synthetic_path, split="train")
            train_dataset = train_dataset.concatenate(synthetic_dataset)
        
        # Tokenize function
        def tokenize_function(examples):
            # Format the prompt
            prompts = []
            for instruction, input_text in zip(examples["instruction"], examples["input"]):
                if input_text:
                    prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endoftext|>\n<|assistant|>\n"
                else:
                    prompt = f"<|user|>\n{instruction}<|endoftext|>\n<|assistant|>\n"
                prompts.append(prompt)
            
            # Tokenize inputs
            tokenized_inputs = self.tokenizer(
                prompts, 
                padding=False, 
                truncation=True,
                max_length=self.config["data_processing"]["max_length"]
            )
            
            # Tokenize outputs and combine with inputs
            tokenized_outputs = self.tokenizer(
                examples["output"],
                padding=False,
                truncation=True,
                max_length=self.config["data_processing"]["max_length"]
            )
            
            # Combine input and output tokens
            input_ids = []
            attention_mask = []
            labels = []
            
            for i in range(len(tokenized_inputs["input_ids"])):
                input_len = len(tokenized_inputs["input_ids"][i])
                output_len = len(tokenized_outputs["input_ids"][i])
                
                # Full sequence: input + output
                full_input_ids = tokenized_inputs["input_ids"][i] + tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]
                full_attention_mask = tokenized_inputs["attention_mask"][i] + tokenized_outputs["attention_mask"][i] + [1]
                
                # Labels: -100 for input tokens (ignored in loss), output tokens for the rest
                full_labels = [-100] * input_len + tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]
                
                input_ids.append(full_input_ids)
                attention_mask.append(full_attention_mask)
                labels.append(full_labels)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # Apply tokenization
        tokenized_train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        
        tokenized_val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
        )
        
        return tokenized_train_dataset, tokenized_val_dataset
    
    def train(self, train_dataset, val_dataset):
        """Train the model.
        
        Args:
            train_dataset: The training dataset.
            val_dataset: The validation dataset.
            
        Returns:
            The trained model.
        """
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=self.config["training"]["learning_rate"],
            num_train_epochs=3,
            max_steps=self.config["training"]["max_steps"],
            warmup_steps=self.config["training"]["warmup_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            evaluation_strategy="steps",
            eval_steps=self.config["training"]["eval_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
            remove_unused_columns=False,
            push_to_hub=False,
            label_names=["labels"],
            lr_scheduler_type=self.config["training"]["lr_scheduler_type"],
            weight_decay=self.config["training"]["weight_decay"],
            max_grad_norm=self.config["training"]["max_grad_norm"],
            fp16=True,
            seed=self.config["training"]["seed"],
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Resume from checkpoint if available
        last_checkpoint = get_last_checkpoint(self.output_dir)
        resume_from_checkpoint = last_checkpoint if last_checkpoint else None
        
        # Start training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the model
        self.model.save_pretrained(os.path.join(self.output_dir, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_model"))
        
        return self.model
    
    def run_training_pipeline(self, train_path: str, val_path: str, synthetic_path: Optional[str] = None):
        """Run the complete training pipeline.
        
        Args:
            train_path: Path to the training dataset.
            val_path: Path to the validation dataset.
            synthetic_path: Path to the synthetic dataset.
            
        Returns:
            The trained model.
        """
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        train_dataset, val_dataset = self.prepare_dataset(train_path, val_path, synthetic_path)
        
        # Train model
        trained_model = self.train(train_dataset, val_dataset)
        
        return trained_model


def train_model(
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    synthetic_path: Optional[str] = None,
    config_path: Optional[str] = None
):
    """Train the model with the specified datasets.
    
    Args:
        train_path: Path to the training dataset. If None, uses the default path.
        val_path: Path to the validation dataset. If None, uses the default path.
        synthetic_path: Path to the synthetic dataset. If None, uses the default path.
        config_path: Path to the configuration file. If None, uses the default config.
        
    Returns:
        The trained model.
    """
    # Set default paths if not provided
    if train_path is None:
        train_path = os.path.join(get_project_root(), "data", "processed", "train.json")
    
    if val_path is None:
        val_path = os.path.join(get_project_root(), "data", "processed", "validation.json")
    
    if synthetic_path is None:
        synthetic_path = os.path.join(get_project_root(), "data", "synthetic", "synthetic_examples.json")
        if not os.path.exists(synthetic_path):
            synthetic_path = None
    
    # Create trainer and run pipeline
    trainer = GraniteTrainer(config_path)
    model = trainer.run_training_pipeline(train_path, val_path, synthetic_path)
    
    return model 