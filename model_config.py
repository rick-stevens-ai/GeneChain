#!/usr/bin/env python3
"""
Model Server Configuration Module
================================

This module loads and provides access to model server configurations defined in model_servers.yaml.
It handles environment variable substitution and provides functions to retrieve server details by
model name or shortname.

Usage:
------
from model_config import get_model_config

# Get by model name
config = get_model_config("gpt-4.1")
print(config)  # {'server': 'api.openai.com', 'shortname': 'gpt41', ...}

# Get by shortname
config = get_model_config("gpt41", by_shortname=True)
print(config)  # {'server': 'api.openai.com', 'shortname': 'gpt41', ...}

# Configure OpenAI client with retrieved configuration
import openai
model_config = get_model_config("gpt-4.1")
openai.api_key = model_config["openai_api_key"]
openai.api_base = model_config["openai_api_base"]
response = openai.ChatCompletion.create(
    model=model_config["openai_model"],
    messages=[...]
)
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Try to import yaml, but provide fallback to json if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("[warn] PyYAML not installed. Using JSON fallback or install with: pip install pyyaml")

# Path to model servers configuration file
CONFIG_FILE = Path("model_servers.yaml")
# Fallbacks in order of preference
CONFIG_FILES = [
    Path("model_servers.yaml"),
    Path("model_servers.yml"),
    Path("model_servers.json")
]
DEFAULT_CONFIG_FILE = CONFIG_FILES[0]  # First choice is YAML

# Cache for loaded configuration 
_config_cache = None

def _substitute_env_vars(value: str) -> str:
    """Replace environment variable references (${VAR_NAME}) with their values."""
    if not isinstance(value, str):
        return value
        
    pattern = r'\${([A-Za-z0-9_]+)}'
    
    def replace_env_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, f"${{{var_name}}}")
    
    return re.sub(pattern, replace_env_var, value)

def load_config(config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load the model servers configuration from the YAML or JSON file.
    
    Args:
        config_file: Optional path to the configuration file. If None, uses the default.
    
    Returns:
        The loaded configuration with environment variables substituted.
    
    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the file cannot be parsed.
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache is not None:
        return _config_cache
    
    file_path = Path(config_file) if config_file else CONFIG_FILE
    
    # Try each fallback file if the specified one doesn't exist
    if not file_path.exists():
        if file_path not in CONFIG_FILES:
            print(f"[warn] Config file {file_path} not found, trying fallbacks")
            found = False
            for fallback in CONFIG_FILES:
                if fallback.exists():
                    print(f"[info] Using fallback config file: {fallback}")
                    file_path = fallback
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"No model server configuration files found. Tried: {file_path} and {CONFIG_FILES}")
        else:
            # Try other fallbacks in CONFIG_FILES
            found = False
            for fallback in CONFIG_FILES:
                if fallback.exists() and fallback != file_path:
                    print(f"[info] Using fallback config file: {fallback}")
                    file_path = fallback
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"No model server configuration files found. Tried: {CONFIG_FILES}")
    
    # Try loading the configuration file
    try:
        with open(file_path, 'r') as f:
            # If PyYAML is available and file has YAML extension, try YAML first
            if YAML_AVAILABLE and file_path.suffix.lower() in ('.yaml', '.yml'):
                config = yaml.safe_load(f)
            else:
                # Otherwise try JSON
                content = f.read()
                config = json.loads(content)
    except json.JSONDecodeError:
        # If JSON parsing fails and we're looking at a YAML file without PyYAML
        if not YAML_AVAILABLE and file_path.suffix.lower() in ('.yaml', '.yml'):
            # Try to find JSON fallback
            for json_file in [p for p in CONFIG_FILES if p.suffix.lower() == '.json']:
                if json_file.exists():
                    print(f"[info] YAML parsing failed, using JSON fallback: {json_file}")
                    with open(json_file, 'r') as f:
                        config = json.loads(f.read())
                        break
            else:
                raise ValueError(f"Could not parse {file_path} as JSON and PyYAML is not installed. "
                               f"Please install PyYAML or provide a valid {file_path.with_suffix('.json')} file.")
        else:
            raise ValueError(f"Could not parse {file_path} as JSON or YAML")
    
    # Process environment variables
    if "servers" in config:
        for server in config["servers"]:
            for key, value in server.items():
                server[key] = _substitute_env_vars(value)
    
    _config_cache = config
    return config

def get_model_config(model_name: str, by_shortname: bool = False, 
                    config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model server by model name or shortname.
    
    Args:
        model_name: The name of the model or shortname to find.
        by_shortname: If True, search by shortname instead of model name.
        config_file: Optional path to the configuration file. If None, uses the default.
    
    Returns:
        The model server configuration dict.
    
    Raises:
        ValueError: If the model is not found in the configuration.
    """
    config = load_config(config_file)
    
    if "servers" not in config:
        raise ValueError("Invalid configuration file: 'servers' key not found")
    
    search_key = "shortname" if by_shortname else "openai_model"
    
    for server in config["servers"]:
        if server.get(search_key) == model_name:
            return server
    
    # If the model wasn't found, check if any model contains the requested name
    # This is useful for partial matches like "gpt-4" matching "gpt-4.1"
    for server in config["servers"]:
        if search_key in server and model_name in server[search_key]:
            return server
    
    # If still not found, raise an error
    raise ValueError(f"Model '{model_name}' not found in configuration")

def list_available_models(config_file: Optional[Union[str, Path]] = None) -> List[Dict[str, str]]:
    """
    List all available models in the configuration.
    
    Args:
        config_file: Optional path to the configuration file. If None, uses the default.
    
    Returns:
        List of dicts with model name and shortname for each server.
    """
    config = load_config(config_file)
    
    if "servers" not in config:
        return []
    
    return [
        {"model": server.get("openai_model", "unknown"), 
         "shortname": server.get("shortname", "unknown")}
        for server in config["servers"]
    ]

def create_openai_client(model_name: str, by_shortname: bool = False,
                         config_file: Optional[Union[str, Path]] = None) -> Any:
    """
    Create an OpenAI client configured for the specified model.
    
    Args:
        model_name: The name of the model or shortname to find.
        by_shortname: If True, search by shortname instead of model name.
        config_file: Optional path to the configuration file. If None, uses the default.
    
    Returns:
        An openai.Client instance configured for the specified model.
    
    Raises:
        ValueError: If the model is not found in the configuration.
        ImportError: If the openai package is not installed.
    """
    try:
        import openai
    except ImportError:
        raise ImportError("The 'openai' package is required to use this function")
    
    config = get_model_config(model_name, by_shortname, config_file)
    
    # The Client-based API (openai>=1.0.0)
    if hasattr(openai, 'Client'):
        return openai.Client(
            api_key=config["openai_api_key"],
            base_url=config["openai_api_base"]
        )
    
    # The module-based API (openai<1.0.0)
    else:
        # Configure the global API settings
        openai.api_key = config["openai_api_key"]
        openai.api_base = config["openai_api_base"]
        return openai

if __name__ == "__main__":
    # Example usage when run as a script
    import argparse
    
    parser = argparse.ArgumentParser(description="Model configuration utility")
    parser.add_argument("--config", dest="config_file", 
                        help="Path to model configuration file (default: model_servers.yaml)")
    parser.add_argument("--list", action="store_true", 
                        help="List all available models")
    parser.add_argument("--show", dest="model_name", 
                        help="Show configuration for a specific model")
    parser.add_argument("--by-shortname", action="store_true",
                        help="Interpret model name as shortname")
    args = parser.parse_args()
    
    try:
        if args.list or (not args.model_name):
            print("Available models:")
            for model in list_available_models(args.config_file):
                print(f"  - {model['model']} (shortname: {model['shortname']})")
        
        if args.model_name:
            # Show configuration for specific model
            model_config = get_model_config(args.model_name, by_shortname=args.by_shortname, config_file=args.config_file)
            print(f"\nConfiguration for {args.model_name}:")
            for key, value in model_config.items():
                if key == "openai_api_key" and value not in (None, "no_key", "${OPENAI_API_KEY}"):
                    print(f"  {key}: [REDACTED]")
                else:
                    print(f"  {key}: {value}")
        elif not args.list:
            # Default behavior - show example with gpt-4.1
            gpt4_config = get_model_config("gpt-4.1", config_file=args.config_file)
            print(f"\nGPT-4.1 configuration:")
            for key, value in gpt4_config.items():
                if key == "openai_api_key" and value not in (None, "no_key", "${OPENAI_API_KEY}"):
                    print(f"  {key}: [REDACTED]")
                else:
                    print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error: {e}")