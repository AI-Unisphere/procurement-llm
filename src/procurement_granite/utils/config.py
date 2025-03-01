"""Configuration utilities for the procurement-granite project."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: The project root directory.
    """
    # This assumes the config.py file is in src/procurement_granite/utils/
    return Path(__file__).parent.parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default config.
        
    Returns:
        Dict[str, Any]: The configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    if config_path is None:
        config_path = os.path.join(get_project_root(), "config", "default_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(data_type: str = "raw") -> Path:
    """Get the path to a data directory.
    
    Args:
        data_type: Type of data directory ('raw', 'processed', or 'synthetic').
        
    Returns:
        Path: The path to the data directory.
        
    Raises:
        ValueError: If data_type is not one of 'raw', 'processed', or 'synthetic'.
    """
    if data_type not in ["raw", "processed", "synthetic"]:
        raise ValueError(f"Invalid data type: {data_type}. Must be one of 'raw', 'processed', or 'synthetic'.")
    
    config = load_config()
    data_path = os.path.join(get_project_root(), config["paths"]["data"][data_type])
    
    return Path(data_path) 