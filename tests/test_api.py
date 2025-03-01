"""Tests for the API module."""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from procurement_granite.api.app import app, get_inference_model, get_document_processor


class TestAPI(unittest.TestCase):
    """Tests for the API module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        
        # Mock the inference model
        self.mock_inference = MagicMock()
        self.mock_inference.extract_rfp_requirements.return_value = "Mock requirements"
        self.mock_inference.identify_evaluation_criteria.return_value = "Mock criteria"
        self.mock_inference.extract_technical_specifications.return_value = "Mock specifications"
        self.mock_inference.score_bid.return_value = "Mock score"
        self.mock_inference.identify_strengths_weaknesses.return_value = "Mock strengths and weaknesses"
        
        # Mock the document processor
        self.mock_processor = MagicMock()
        self.mock_processor.supported_formats = ["pdf", "docx", "txt"]
        self.mock_processor.process_document.return_value = "Mock document text"
        
        # Patch the dependency injection functions
        self.get_inference_patch = patch("procurement_granite.api.app.get_inference_model", return_value=self.mock_inference)
        self.get_processor_patch = patch("procurement_granite.api.app.get_document_processor", return_value=self.mock_processor)
        
        self.get_inference_patch.start()
        self.get_processor_patch.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.get_inference_patch.stop()
        self.get_processor_patch.stop()
    
    def test_root(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Welcome to the Procurement Document Analysis API"})
    
    def test_analyze_rfp(self):
        """Test the analyze-rfp endpoint."""
        response = self.client.post(
            "/analyze-rfp",
            json={"text": "Test RFP text"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "requirements": "Mock requirements",
                "evaluation_criteria": "Mock criteria",
                "technical_specifications": "Mock specifications",
            },
        )
        self.mock_inference.extract_rfp_requirements.assert_called_once_with("Test RFP text")
        self.mock_inference.identify_evaluation_criteria.assert_called_once_with("Test RFP text")
        self.mock_inference.extract_technical_specifications.assert_called_once_with("Test RFP text")
    
    def test_evaluate_bid(self):
        """Test the evaluate-bid endpoint."""
        response = self.client.post(
            "/evaluate-bid",
            json={"rfp_text": "Test RFP text", "bid_text": "Test bid text"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "score": "Mock score",
                "strengths_weaknesses": "Mock strengths and weaknesses",
            },
        )
        self.mock_inference.score_bid.assert_called_once_with("Test RFP text", "Test bid text")
        self.mock_inference.identify_strengths_weaknesses.assert_called_once_with("Test RFP text", "Test bid text")
    
    @patch("procurement_granite.api.app.os.makedirs")
    @patch("procurement_granite.api.app.os.remove")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("procurement_granite.api.app.Path")
    def test_upload_rfp(self, mock_path, mock_open, mock_remove, mock_makedirs):
        """Test the upload-rfp endpoint."""
        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.suffix = ".txt"
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = mock_path_instance
        
        # Create a test file
        test_file_content = b"Test RFP content"
        
        # Send the request
        response = self.client.post(
            "/upload-rfp",
            files={"file": ("test.txt", test_file_content, "text/plain")},
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "filename": "test.txt",
                "requirements": "Mock requirements",
                "evaluation_criteria": "Mock criteria",
                "technical_specifications": "Mock specifications",
            },
        )
        
        # Check that the file was processed
        self.mock_processor.process_document.assert_called_once()
        
        # Check that the model was called
        self.mock_inference.extract_rfp_requirements.assert_called_once_with("Mock document text")
        self.mock_inference.identify_evaluation_criteria.assert_called_once_with("Mock document text")
        self.mock_inference.extract_technical_specifications.assert_called_once_with("Mock document text")
    
    @patch("procurement_granite.api.app.os.makedirs")
    @patch("procurement_granite.api.app.os.remove")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("procurement_granite.api.app.Path")
    def test_upload_bid(self, mock_path, mock_open, mock_remove, mock_makedirs):
        """Test the upload-bid endpoint."""
        # Mock Path
        mock_path_instance = MagicMock()
        mock_path_instance.suffix = ".txt"
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__.return_value = mock_path_instance
        
        # Create test files
        test_rfp_content = b"Test RFP content"
        test_bid_content = b"Test bid content"
        
        # Send the request
        response = self.client.post(
            "/upload-bid",
            files={
                "rfp_file": ("rfp.txt", test_rfp_content, "text/plain"),
                "bid_file": ("bid.txt", test_bid_content, "text/plain"),
            },
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "rfp_filename": "rfp.txt",
                "bid_filename": "bid.txt",
                "score": "Mock score",
                "strengths_weaknesses": "Mock strengths and weaknesses",
            },
        )
        
        # Check that the files were processed
        self.assertEqual(self.mock_processor.process_document.call_count, 2)
        
        # Check that the model was called
        self.mock_inference.score_bid.assert_called_once_with("Mock document text", "Mock document text")
        self.mock_inference.identify_strengths_weaknesses.assert_called_once_with("Mock document text", "Mock document text")


if __name__ == "__main__":
    unittest.main() 