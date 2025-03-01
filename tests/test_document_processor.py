"""Tests for the document processor module."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from procurement_granite.data.document_processor import DocumentProcessor
from procurement_granite.utils.config import get_project_root


class TestDocumentProcessor(unittest.TestCase):
    """Tests for the document processor module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
        self.test_dir = Path(get_project_root()) / "tests" / "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        for file_path in self.test_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    
    def test_init(self):
        """Test initialization."""
        self.assertIsInstance(self.processor, DocumentProcessor)
        self.assertIn("pdf", self.processor.supported_formats)
        self.assertIn("docx", self.processor.supported_formats)
        self.assertIn("txt", self.processor.supported_formats)
    
    @patch("builtins.open", new_callable=mock_open, read_data="Test content")
    def test_process_txt(self, mock_file):
        """Test processing a TXT file."""
        # Create a test TXT file
        test_file = self.test_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # Process the file
        text = self.processor._process_txt(test_file)
        
        # Check the result
        self.assertEqual(text, "Test content")
    
    @patch("PyPDF2.PdfReader")
    def test_process_pdf(self, mock_pdf_reader):
        """Test processing a PDF file."""
        # Mock PDF reader
        mock_page = unittest.mock.MagicMock()
        mock_page.extract_text.return_value = "Test PDF content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        # Create a test PDF file path
        test_file = self.test_dir / "test.pdf"
        
        # Mock the file existence
        with patch("pathlib.Path.exists", return_value=True):
            # Process the file
            with patch("builtins.open", mock_open()):
                text = self.processor._process_pdf(test_file)
        
        # Check the result
        self.assertEqual(text, "Test PDF content\n\n")
    
    @patch("docx.Document")
    def test_process_docx(self, mock_document):
        """Test processing a DOCX file."""
        # Mock Document
        mock_doc = unittest.mock.MagicMock()
        mock_paragraph1 = unittest.mock.MagicMock()
        mock_paragraph1.text = "Test DOCX paragraph 1"
        mock_paragraph2 = unittest.mock.MagicMock()
        mock_paragraph2.text = "Test DOCX paragraph 2"
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
        mock_doc.tables = []
        mock_document.return_value = mock_doc
        
        # Create a test DOCX file path
        test_file = self.test_dir / "test.docx"
        
        # Process the file
        text = self.processor._process_docx(test_file)
        
        # Check the result
        self.assertEqual(text, "Test DOCX paragraph 1\nTest DOCX paragraph 2\n")
    
    def test_unsupported_format(self):
        """Test processing an unsupported file format."""
        # Create a test file with unsupported extension
        test_file = self.test_dir / "test.xyz"
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # Process the file
        with self.assertRaises(ValueError):
            self.processor.process_document(test_file)
    
    def test_file_not_found(self):
        """Test processing a non-existent file."""
        # Create a non-existent file path
        test_file = self.test_dir / "nonexistent.txt"
        
        # Process the file
        with self.assertRaises(FileNotFoundError):
            self.processor.process_document(test_file)


if __name__ == "__main__":
    unittest.main() 