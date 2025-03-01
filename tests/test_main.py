"""Tests for the main module."""

import unittest
from unittest.mock import patch, MagicMock

from procurement_granite.main import (
    process_data,
    train,
    evaluate,
    serve,
    run_pipeline,
    main,
)


class TestMain(unittest.TestCase):
    """Tests for the main module."""
    
    def test_process_data(self):
        """Test process_data function."""
        with patch("procurement_granite.main.process_raw_documents") as mock_process_raw:
            with patch("procurement_granite.main.create_datasets_from_processed_documents") as mock_create_datasets:
                # Mock return values
                mock_process_raw.return_value = {"doc1.txt": "content1", "doc2.txt": "content2"}
                mock_create_datasets.return_value = (MagicMock(), MagicMock(), MagicMock())
                
                # Call the function
                process_data()
                
                # Check that the functions were called
                mock_process_raw.assert_called_once()
                mock_create_datasets.assert_called_once_with(
                    create_synthetic=True,
                    num_synthetic=100,
                )
    
    def test_train(self):
        """Test train function."""
        with patch("procurement_granite.main.train_model") as mock_train_model:
            # Mock return value
            mock_train_model.return_value = MagicMock()
            
            # Call the function
            train()
            
            # Check that the function was called
            mock_train_model.assert_called_once_with(config_path=None)
    
    def test_evaluate(self):
        """Test evaluate function."""
        with patch("procurement_granite.main.evaluate_model_on_test_data") as mock_evaluate:
            # Mock return value
            mock_evaluate.return_value = {
                "task1": {"metric1": 0.8, "metric2": 0.9},
                "task2": {"metric1": 0.7, "metric2": 0.8},
                "overall": {"metric1": 0.75, "metric2": 0.85},
            }
            
            # Call the function
            evaluate()
            
            # Check that the function was called
            mock_evaluate.assert_called_once_with(
                model_path=None,
                config_path=None,
            )
    
    def test_serve(self):
        """Test serve function."""
        with patch("procurement_granite.main.start_server") as mock_start_server:
            # Call the function
            serve()
            
            # Check that the function was called
            mock_start_server.assert_called_once()
    
    def test_run_pipeline(self):
        """Test run_pipeline function."""
        with patch("procurement_granite.main.process_data") as mock_process:
            with patch("procurement_granite.main.train") as mock_train:
                with patch("procurement_granite.main.evaluate") as mock_evaluate:
                    with patch("procurement_granite.main.serve") as mock_serve:
                        # Call the function
                        run_pipeline()
                        
                        # Check that the functions were called
                        mock_process.assert_called_once_with(None)
                        mock_train.assert_called_once_with(None)
                        mock_evaluate.assert_called_once_with(config_path=None)
                        mock_serve.assert_called_once_with(config_path=None)
    
    @patch("procurement_granite.main.argparse.ArgumentParser")
    def test_main_process(self, mock_argparse):
        """Test main function with process command."""
        # Mock ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock parse_args
        mock_args = MagicMock()
        mock_args.command = "process"
        mock_args.config = "config.yaml"
        mock_parser.parse_args.return_value = mock_args
        
        # Mock process_data
        with patch("procurement_granite.main.process_data") as mock_process:
            # Call the function
            main()
            
            # Check that process_data was called
            mock_process.assert_called_once_with("config.yaml")
    
    @patch("procurement_granite.main.argparse.ArgumentParser")
    def test_main_train(self, mock_argparse):
        """Test main function with train command."""
        # Mock ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock parse_args
        mock_args = MagicMock()
        mock_args.command = "train"
        mock_args.config = "config.yaml"
        mock_parser.parse_args.return_value = mock_args
        
        # Mock train
        with patch("procurement_granite.main.train") as mock_train:
            # Call the function
            main()
            
            # Check that train was called
            mock_train.assert_called_once_with("config.yaml")
    
    @patch("procurement_granite.main.argparse.ArgumentParser")
    def test_main_evaluate(self, mock_argparse):
        """Test main function with evaluate command."""
        # Mock ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock parse_args
        mock_args = MagicMock()
        mock_args.command = "evaluate"
        mock_args.model = "model_path"
        mock_args.config = "config.yaml"
        mock_parser.parse_args.return_value = mock_args
        
        # Mock evaluate
        with patch("procurement_granite.main.evaluate") as mock_evaluate:
            # Call the function
            main()
            
            # Check that evaluate was called
            mock_evaluate.assert_called_once_with("model_path", "config.yaml")
    
    @patch("procurement_granite.main.argparse.ArgumentParser")
    def test_main_serve(self, mock_argparse):
        """Test main function with serve command."""
        # Mock ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock parse_args
        mock_args = MagicMock()
        mock_args.command = "serve"
        mock_args.model = "model_path"
        mock_args.config = "config.yaml"
        mock_parser.parse_args.return_value = mock_args
        
        # Mock serve
        with patch("procurement_granite.main.serve") as mock_serve:
            # Call the function
            main()
            
            # Check that serve was called
            mock_serve.assert_called_once_with("model_path", "config.yaml")
    
    @patch("procurement_granite.main.argparse.ArgumentParser")
    def test_main_pipeline(self, mock_argparse):
        """Test main function with pipeline command."""
        # Mock ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock parse_args
        mock_args = MagicMock()
        mock_args.command = "pipeline"
        mock_args.config = "config.yaml"
        mock_parser.parse_args.return_value = mock_args
        
        # Mock run_pipeline
        with patch("procurement_granite.main.run_pipeline") as mock_run_pipeline:
            # Call the function
            main()
            
            # Check that run_pipeline was called
            mock_run_pipeline.assert_called_once_with("config.yaml")
    
    @patch("procurement_granite.main.argparse.ArgumentParser")
    def test_main_no_command(self, mock_argparse):
        """Test main function with no command."""
        # Mock ArgumentParser
        mock_parser = MagicMock()
        mock_argparse.return_value = mock_parser
        
        # Mock parse_args
        mock_args = MagicMock()
        mock_args.command = None
        mock_parser.parse_args.return_value = mock_args
        
        # Call the function
        main()
        
        # Check that print_help was called
        mock_parser.print_help.assert_called_once()


if __name__ == "__main__":
    unittest.main() 