"""Tests for the inference module."""

import unittest
from unittest.mock import patch, MagicMock

from procurement_granite.model.inference import GraniteInference, load_inference_model


class TestInference(unittest.TestCase):
    """Tests for the inference module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the config
        self.mock_config = {
            "paths": {
                "models": "models",
            },
            "model": {
                "base_model": "ibm/granite-3.2-8b-instruct",
                "quantization": None,
            },
        }
        
        # Patch the config loading
        self.config_patch = patch("procurement_granite.model.inference.load_config", return_value=self.mock_config)
        self.config_patch.start()
        
        # Mock the model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_pipeline = MagicMock()
        
        # Mock the pipeline output
        self.mock_pipeline.return_value = [{"generated_text": "<|user|>\nTest instruction<|endoftext|>\n<|assistant|>\nTest response"}]
        
        # Patch the model loading
        self.model_patch = patch("procurement_granite.model.inference.AutoModelForCausalLM.from_pretrained", return_value=self.mock_model)
        self.tokenizer_patch = patch("procurement_granite.model.inference.AutoTokenizer.from_pretrained", return_value=self.mock_tokenizer)
        self.peft_patch = patch("procurement_granite.model.inference.PeftModel.from_pretrained", return_value=self.mock_model)
        self.pipeline_patch = patch("procurement_granite.model.inference.pipeline", return_value=self.mock_pipeline)
        
        self.model_patch.start()
        self.tokenizer_patch.start()
        self.peft_patch.start()
        self.pipeline_patch.start()
        
        # Create the inference object
        self.inference = GraniteInference(model_path="test_model_path")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.config_patch.stop()
        self.model_patch.stop()
        self.tokenizer_patch.stop()
        self.peft_patch.stop()
        self.pipeline_patch.stop()
    
    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.inference, GraniteInference)
        self.assertEqual(self.inference.model_path, "test_model_path")
        self.assertEqual(self.inference.device, "cuda" if unittest.mock.patch("torch.cuda.is_available", return_value=True) else "cpu")
    
    def test_load_model(self):
        """Test loading the model."""
        model, tokenizer = self.inference.load_model()
        
        self.assertEqual(model, self.mock_model)
        self.assertEqual(tokenizer, self.mock_tokenizer)
        self.assertEqual(self.inference.pipeline, self.mock_pipeline)
    
    def test_generate(self):
        """Test generating text."""
        # Load the model
        self.inference.load_model()
        
        # Generate text
        response = self.inference.generate("Test instruction")
        
        # Check the result
        self.assertEqual(response, "Test response")
        
        # Check that the pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_extract_rfp_requirements(self):
        """Test extracting RFP requirements."""
        # Load the model
        self.inference.load_model()
        
        # Extract requirements
        response = self.inference.extract_rfp_requirements("Test RFP text")
        
        # Check the result
        self.assertEqual(response, "Test response")
        
        # Check that the pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_identify_evaluation_criteria(self):
        """Test identifying evaluation criteria."""
        # Load the model
        self.inference.load_model()
        
        # Identify criteria
        response = self.inference.identify_evaluation_criteria("Test RFP text")
        
        # Check the result
        self.assertEqual(response, "Test response")
        
        # Check that the pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_extract_technical_specifications(self):
        """Test extracting technical specifications."""
        # Load the model
        self.inference.load_model()
        
        # Extract specifications
        response = self.inference.extract_technical_specifications("Test RFP text")
        
        # Check the result
        self.assertEqual(response, "Test response")
        
        # Check that the pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_score_bid(self):
        """Test scoring a bid."""
        # Load the model
        self.inference.load_model()
        
        # Score bid
        response = self.inference.score_bid("Test RFP text", "Test bid text")
        
        # Check the result
        self.assertEqual(response, "Test response")
        
        # Check that the pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_identify_strengths_weaknesses(self):
        """Test identifying strengths and weaknesses."""
        # Load the model
        self.inference.load_model()
        
        # Identify strengths and weaknesses
        response = self.inference.identify_strengths_weaknesses("Test RFP text", "Test bid text")
        
        # Check the result
        self.assertEqual(response, "Test response")
        
        # Check that the pipeline was called
        self.mock_pipeline.assert_called_once()
    
    def test_load_inference_model(self):
        """Test loading the inference model."""
        with patch("procurement_granite.model.inference.GraniteInference") as mock_inference_class:
            mock_inference = MagicMock()
            mock_inference_class.return_value = mock_inference
            
            # Load the inference model
            inference = load_inference_model("test_model_path", "test_config_path", "test_device")
            
            # Check that the inference model was created and loaded
            mock_inference_class.assert_called_once_with("test_model_path", "test_config_path", "test_device")
            mock_inference.load_model.assert_called_once()


if __name__ == "__main__":
    unittest.main() 