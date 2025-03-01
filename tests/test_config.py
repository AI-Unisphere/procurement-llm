"""Tests for the configuration module."""

import os
import unittest
from pathlib import Path

from procurement_granite.utils.config import get_project_root, load_config, get_data_path


class TestConfig(unittest.TestCase):
    """Tests for the configuration module."""
    
    def test_get_project_root(self):
        """Test get_project_root function."""
        root = get_project_root()
        self.assertTrue(isinstance(root, Path))
        self.assertTrue(root.exists())
        self.assertTrue((root / "config").exists())
    
    def test_load_config(self):
        """Test load_config function."""
        config = load_config()
        self.assertTrue(isinstance(config, dict))
        self.assertIn("paths", config)
        self.assertIn("model", config)
        self.assertIn("training", config)
        self.assertIn("data_processing", config)
        self.assertIn("evaluation", config)
        self.assertIn("api", config)
        self.assertIn("deployment", config)
    
    def test_get_data_path(self):
        """Test get_data_path function."""
        # Test raw data path
        raw_path = get_data_path("raw")
        self.assertTrue(isinstance(raw_path, Path))
        self.assertEqual(raw_path.name, "raw")
        
        # Test processed data path
        processed_path = get_data_path("processed")
        self.assertTrue(isinstance(processed_path, Path))
        self.assertEqual(processed_path.name, "processed")
        
        # Test synthetic data path
        synthetic_path = get_data_path("synthetic")
        self.assertTrue(isinstance(synthetic_path, Path))
        self.assertEqual(synthetic_path.name, "synthetic")
        
        # Test invalid data type
        with self.assertRaises(ValueError):
            get_data_path("invalid")


if __name__ == "__main__":
    unittest.main() 