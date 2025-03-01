"""Inference utilities for the procurement-granite project."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from procurement_granite.utils.config import get_project_root, load_config


class GraniteInference:
    """Inference for the fine-tuned IBM GRANITE 3.2 8B Instruct model."""
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the inference module.
        
        Args:
            model_path: Path to the fine-tuned model. If None, uses the default path.
            config_path: Path to the configuration file. If None, uses the default config.
            device: Device to run inference on. If None, uses CUDA if available, otherwise CPU.
        """
        self.config = load_config(config_path)
        
        if model_path is None:
            model_path = os.path.join(
                get_project_root(), 
                self.config["paths"]["models"], 
                "final_model"
            )
        
        self.model_path = model_path
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
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
        
        # Load base model with quantization if specified
        base_model_name = self.config["model"]["base_model"]
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Load the fine-tuned LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_path,
            device_map="auto",
        )
        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        
        return self.model, self.tokenizer
    
    def generate(
        self, 
        instruction: str, 
        input_text: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """Generate text using the fine-tuned model.
        
        Args:
            instruction: The instruction for the model.
            input_text: The input text for the model. If None, only the instruction is used.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature for sampling.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            repetition_penalty: Repetition penalty parameter.
            do_sample: Whether to use sampling or greedy decoding.
            
        Returns:
            str: The generated text.
        """
        if self.pipeline is None:
            self.load_model()
        
        # Format the prompt
        if input_text:
            prompt = f"<|user|>\n{instruction}\n\n{input_text}<|endoftext|>\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{instruction}<|endoftext|>\n<|assistant|>\n"
        
        # Generate text
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt from the generated text
        response = generated_text[len(prompt):]
        
        # Remove any trailing assistant tags
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[0]
        
        # Remove any user tags
        if "<|user|>" in response:
            response = response.split("<|user|>")[0]
        
        # Remove any endoftext tags
        if "<|endoftext|>" in response:
            response = response.split("<|endoftext|>")[0]
        
        return response.strip()
    
    def extract_rfp_requirements(self, rfp_text: str) -> str:
        """Extract key requirements from an RFP document.
        
        Args:
            rfp_text: The text of the RFP document.
            
        Returns:
            str: The extracted requirements.
        """
        instruction = "Extract the key requirements from this RFP document."
        return self.generate(instruction, rfp_text)
    
    def identify_evaluation_criteria(self, rfp_text: str) -> str:
        """Identify evaluation criteria from an RFP document.
        
        Args:
            rfp_text: The text of the RFP document.
            
        Returns:
            str: The identified evaluation criteria.
        """
        instruction = "Identify the evaluation criteria and their weights from this RFP document."
        return self.generate(instruction, rfp_text)
    
    def extract_technical_specifications(self, rfp_text: str) -> str:
        """Extract technical specifications from an RFP document.
        
        Args:
            rfp_text: The text of the RFP document.
            
        Returns:
            str: The extracted technical specifications.
        """
        instruction = "Extract the technical specifications for the required connectivity services from this RFP document."
        return self.generate(instruction, rfp_text)
    
    def score_bid(self, rfp_text: str, bid_text: str) -> str:
        """Score a bid against RFP requirements.
        
        Args:
            rfp_text: The text of the RFP document.
            bid_text: The text of the bid document.
            
        Returns:
            str: The bid score and justification.
        """
        instruction = "Score this bid against the RFP requirements."
        input_text = f"rfp_text: {rfp_text}\nbid_text: {bid_text}"
        return self.generate(instruction, input_text)
    
    def identify_strengths_weaknesses(self, rfp_text: str, bid_text: str) -> str:
        """Identify strengths and weaknesses of a bid proposal.
        
        Args:
            rfp_text: The text of the RFP document.
            bid_text: The text of the bid document.
            
        Returns:
            str: The identified strengths and weaknesses.
        """
        instruction = "Identify the strengths and weaknesses of this bid proposal against the RFP requirements."
        input_text = f"rfp_text: {rfp_text}\nbid_text: {bid_text}"
        return self.generate(instruction, input_text)


def load_inference_model(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
) -> GraniteInference:
    """Load the inference model.
    
    Args:
        model_path: Path to the fine-tuned model. If None, uses the default path.
        config_path: Path to the configuration file. If None, uses the default config.
        device: Device to run inference on. If None, uses CUDA if available, otherwise CPU.
        
    Returns:
        GraniteInference: The loaded inference model.
    """
    inference = GraniteInference(model_path, config_path, device)
    inference.load_model()
    return inference 