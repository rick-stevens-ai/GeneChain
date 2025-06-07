#!/usr/bin/env python3
"""
Compare Model Predictions for Gene Interactions
===============================================

This script compares predictions made by different models for gene pair interactions:
1. Loads all available models from model_servers.yaml
2. Samples a specified number of interaction files from INTERACTION_CACHE
3. For each gene pair, generates new predictions using each model
4. Compares the predictions to identify agreement and disagreement patterns
5. Outputs a summary report

Usage:
------
$ export OPENAI_API_KEY="sk-..."
$ python compare_model_predictions.py --pairs 20 --out comparison_report.json

Requirements:
------------
- Python ≥3.8
- openai
- All dependencies required by gene_chain_v1.py
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional, Union
from collections import defaultdict
import traceback

# Import model configuration
try:
    from model_config import get_model_config, list_available_models
    MODEL_CONFIG_AVAILABLE = True
except ImportError:
    MODEL_CONFIG_AVAILABLE = False
    print("[warn] model_config module not found; using default OpenAI configuration")

# Import from gene_chain_v1 if possible
try:
    from gene_chain_v1 import gpt_call
    GENE_CHAIN_AVAILABLE = True
except ImportError:
    GENE_CHAIN_AVAILABLE = False
    print("[warn] gene_chain_v1 module not found; using local implementation")

# Define the cache directory for interaction data
INTERACTION_CACHE = Path("INTERACTION_CACHE")
if not INTERACTION_CACHE.exists():
    INTERACTION_CACHE.mkdir(exist_ok=True)

# Try to import optional json5 package for more permissive JSON parsing
try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    JSON5_AVAILABLE = False

# OpenAI implementation if gene_chain not available
if not GENE_CHAIN_AVAILABLE:
    import openai
    
    def gpt_call(messages: List[Dict[str, str]], *, model: str = "gpt-4.1", 
                retries: int = 3, backoff: float = 5.0, config_file: Optional[str] = None) -> str:
        """Robust wrapper around openai.ChatCompletion.create with simple retry."""
        global MODEL_CONFIG_AVAILABLE
        
        # Configure OpenAI client based on model_config if available
        if MODEL_CONFIG_AVAILABLE:
            try:
                # Try to get model configuration by name or shortname
                model_config = None
                try:
                    # First try by model name
                    model_config = get_model_config(model, config_file=config_file)
                except ValueError:
                    # Then try by shortname
                    try:
                        model_config = get_model_config(model, by_shortname=True, config_file=config_file)
                    except ValueError:
                        print(f"[warn] Model '{model}' not found in configuration, using default OpenAI settings")
                
                # Apply configuration if found
                if model_config:
                    # Save the original API settings to restore later if needed
                    original_api_key = openai.api_key
                    original_api_base = getattr(openai, "api_base", None)
                    
                    # Apply the model-specific settings
                    openai.api_key = model_config["openai_api_key"]
                    openai.api_base = model_config["openai_api_base"]
                    model_name = model_config["openai_model"]
                    
                    print(f"[info] Using model configuration for {model}: {model_name} at {openai.api_base}")
                    
                    # Update the model name to use
                    model = model_name
            except Exception as e:
                print(f"[warn] Error configuring model from YAML: {e}. Using default configuration.")
        
        # Make the API call with retries
        for attempt in range(1, retries + 1):
            try:
                # For all models, use minimal required parameters to avoid compatibility issues
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                )
                
                return response.choices[0].message["content"].strip()
            except Exception as exc:
                if attempt == retries:
                    raise
                print(f"[warn] OpenAI call failed on attempt {attempt}: {exc}.  Retrying in {backoff} s…",
                      file=sys.stderr)
                time.sleep(backoff)
        raise RuntimeError("Unreachable code in gpt_call")

def clean_json_string(raw_json: str) -> str:
    """
    Clean up a potentially malformed JSON string.
    Handles various issues like markdown code blocks, quoted JSON, etc.
    Special handling for Qwen-style outputs with markdown fences and multiple wrapping.
    """
    cleaned = raw_json.strip()
    
    # Handle the specific Qwen issue where JSON is wrapped in markdown fences AND quotes
    # This pattern matches markdown fences with potential quote wrapping
    qwen_pattern = r'[\'"]?```(?:json)?\s*(.*?)\s*```[\'"]?'
    qwen_match = re.search(qwen_pattern, cleaned, re.DOTALL)
    if qwen_match:
        inner_content = qwen_match.group(1).strip()
        # Check if the inner content is valid JSON
        try:
            json.loads(inner_content)
            return inner_content  # If valid, return it directly
        except json.JSONDecodeError:
            # If not valid, continue with other cleaning methods
            cleaned = inner_content
    
    # Standard markdown code block removal (if the Qwen pattern didn't match or wasn't valid)
    markdown_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'`(.*?)`'
    ]
    
    for pattern in markdown_patterns:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    
    # Handle multiple layers of quote wrapping (common in Qwen outputs)
    # Try to unwrap up to 3 layers of quotes
    for _ in range(3):
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            try:
                # Try to interpret as a quoted JSON string
                first_char = cleaned[0]
                if first_char == '"':
                    unquoted = json.loads(cleaned)
                else:  # first_char == "'"
                    # Handle single quotes by replacing with double quotes for JSON parsing
                    double_quoted = '"' + cleaned[1:-1].replace('"', '\\"').replace("\\'", "'") + '"'
                    unquoted = json.loads(double_quoted)
                    
                if isinstance(unquoted, str):
                    cleaned = unquoted
                else:
                    # If it parsed to a non-string value, we're done unwrapping
                    break
            except:
                # If parsing fails, stop unwrapping
                break
    
    # Some models (like Qwen) might use single quotes instead of double quotes
    if not cleaned.startswith("{") and not cleaned.startswith("["):
        # Replace single quotes with double quotes, being careful with already escaped quotes
        # This is a simplistic approach - won't handle all edge cases
        cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
        cleaned = cleaned.replace("\\'", "'")  # Fix any escaped single quotes
    
    # Check for and fix common JSON syntax errors
    
    # Fix missing quotes around keys
    cleaned = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', cleaned)
    
    # Fix trailing commas in arrays and objects
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix missing quotes around string values (this is a bit risky but worth trying)
    # only apply to obviously non-quoted strings
    cleaned = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\-\s]+)([,}])', r': "\1"\2', cleaned)
    
    return cleaned

def parse_json_with_fallbacks(raw_json: str, model_id: str = "") -> Dict[str, Any]:
    """
    Parse a JSON string with multiple fallback methods if the initial parse fails.
    If all methods fail, tries to fix the JSON using GPT-4.1.
    """
    # First try direct parsing
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        # Clean up the JSON string and try again
        cleaned = clean_json_string(raw_json)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try a more permissive JSON parser if available
            if JSON5_AVAILABLE:
                try:
                    return json5.loads(cleaned)
                except Exception:
                    pass  # Continue to GPT-4.1 fallback
            
            # If all previous methods fail, use GPT-4.1 to fix the JSON
            try:
                print(f"[info] Attempting to fix malformed JSON from model {model_id} with GPT-4.1")
                fix_prompt = f"""
                The following is a malformed JSON string from a model. Fix the JSON syntax errors to make it valid.
                Only return the fixed JSON, nothing else. No explanations or markdown.
                ```
                {raw_json}
                ```
                """
                messages = [
                    {"role": "system", "content": "You are a helpful tool that fixes malformed JSON. Only respond with the fixed JSON, nothing else."},
                    {"role": "user", "content": fix_prompt}
                ]
                
                # Use GPT-4.1 to fix the JSON
                fixed_json = gpt_call(messages, model="gpt-4.1", config_file=None)
                
                # Clean up the fixed JSON (remove any markdown formatting, etc.)
                fixed_json = clean_json_string(fixed_json)
                
                # Try to parse the fixed JSON
                return json.loads(fixed_json)
                
            except Exception as exc:
                # If all attempts fail, raise a clear error
                raise ValueError(f"Failed to parse JSON even after GPT-4.1 cleanup. Raw output was:\n{raw_json}\n") from exc

def query_paths(entity_a: str, entity_b: str, n_paths: int, model: str, config_file: Optional[str] = None) -> Dict[str, Any]:
    """Ask model for interaction paths between *entity_a* and *entity_b*."""
    sys_prompt = (
        "You are a systems‑biology reasoning engine.  "
        "When asked for interactions between biological entities, you consult the cell‑biology "
        "literature and output strictly‑valid JSON describing plausible mechanistic chains.  "
        "Each chain should include mechanistic verbs (e.g. 'phosphorylates', 'inhibits'), quote "
        "key evidence (PMID or database), and assign a subjective probability ∈ (0,1].  "
        "If no biologically plausible path exists, respond with {\"no_path\": true, \"reason\": string}."
    )
    user_prompt = (
        f"Find up to {n_paths} causal interaction path(s) connecting \"{entity_a}\" and \"{entity_b}\". "
        "Return a JSON object with key 'paths'.  Each path *must* be an object with:\n"
        "  • 'edges'  : list[ {source, target, mechanism, probability, evidence} ]\n"
        "  • 'overall_probability': float\n"
        "  • 'summary'            : string ≤ 40 words\n"
        "Paths should be ordered by ascending length, breaking ties by descending overall_probability.\n"
        "Output nothing except the JSON object. Do not wrap the JSON in markdown fences or quotes."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Since we're having issues with the max_completion_tokens parameter,
    # let's just use the default parameters and let gpt_call handle the model-specific differences
    raw = gpt_call(messages, model=model, config_file=config_file)
        
    try:
        return parse_json_with_fallbacks(raw, model_id=model)
    except ValueError as exc:
        print(f"[error] JSON parsing failed for {model} on {entity_a}-{entity_b}. Error: {str(exc)}")
        return {"error": str(exc), "raw_output": raw}

def test_model_endpoint(model_info: Dict[str, str], timeout: int = 5, config_file: Optional[str] = None) -> bool:
    """Test if a model endpoint is available by making a minimal API call with a simple query."""
    if not MODEL_CONFIG_AVAILABLE:
        # Without model_config, assume the default OpenAI models are available
        return model_info["shortname"] in ["gpt41", "o3"]
    
    import openai
    from urllib.parse import urlparse
    import socket
    
    try:
        # Try to get model configuration
        model_config = None
        try:
            # First try by model name
            model_config = get_model_config(model_info["model"], config_file=config_file)
        except ValueError:
            # Then try by shortname
            try:
                model_config = get_model_config(model_info["shortname"], by_shortname=True, config_file=config_file)
            except ValueError:
                return False
        
        if not model_config:
            return False
        
        # First, quick check if the host is reachable
        api_base = model_config["openai_api_base"]
        try:
            parsed_url = urlparse(api_base)
            hostname = parsed_url.hostname
            # Only do socket test for non-standard endpoints
            if hostname and hostname not in ["api.openai.com", "api.anthropic.com"]:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout/2)
                port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
                result = sock.connect_ex((hostname, port))
                sock.close()
                if result != 0:
                    return False
        except Exception:
            # If socket test fails, continue to API test anyway
            pass
            
        # Save the original API settings to restore later
        original_api_key = openai.api_key
        original_api_base = getattr(openai, "api_base", None)
        
        # Apply the model-specific settings
        openai.api_key = model_config["openai_api_key"]
        openai.api_base = model_config["openai_api_base"]
        model_name = model_config["openai_model"]
        
        # Make a minimal chat completion call with short timeout
        try:
            # Create a very simple prompt
            messages = [
                {"role": "user", "content": "Hi"}
            ]
            
            # For OpenAI v1.0+ API
            if hasattr(openai, 'Client'):
                client = openai.Client(
                    api_key=model_config["openai_api_key"],
                    base_url=model_config["openai_api_base"],
                    timeout=timeout
                )
                # Use a minimal token request
                # Use minimal parameters for all models to avoid compatibility issues
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
            # For OpenAI legacy API
            else:
                openai.api_key = model_config["openai_api_key"]
                openai.api_base = model_config["openai_api_base"]
                
                # Use minimal parameters for all models
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                )
            
            # Restore original settings
            openai.api_key = original_api_key
            if original_api_base:
                openai.api_base = original_api_base
                
            print(f"[info] Successfully tested endpoint for {model_info['shortname']}")
            return True
            
        except Exception as e:
            # Restore original settings
            openai.api_key = original_api_key
            if original_api_base:
                openai.api_base = original_api_base
                
            print(f"[debug] Endpoint test failed for {model_info['shortname']}: {str(e)}")
            return False
            
    except Exception as e:
        print(f"[debug] Error testing model endpoint {model_info['shortname']}: {str(e)}")
        return False

def get_available_models(config_file: Optional[str] = None) -> List[Dict[str, str]]:
    """Get a list of all available models in the configuration."""
    if not MODEL_CONFIG_AVAILABLE:
        # Default to a basic model set if model_config is not available
        return [
            {"model": "gpt-4.1", "shortname": "gpt41"},
            {"model": "gpt-4o", "shortname": "o3"}
        ]
    
    try:
        all_models = list_available_models(config_file)
        return all_models
    except Exception as e:
        print(f"[warn] Error listing available models: {e}. Using default models.")
        return [
            {"model": "gpt-4.1", "shortname": "gpt41"},
            {"model": "gpt-4o", "shortname": "o3"}
        ]

def find_interaction_files() -> List[Path]:
    """Find all interaction files in the INTERACTION_CACHE directory."""
    # Look for files matching the network_*_*_interactions.json pattern
    pattern = os.path.join(str(INTERACTION_CACHE), "network_*_*_interactions.json")
    file_paths = glob.glob(pattern)
    
    # Also check the current directory for interaction files
    local_pattern = "network_*_*_interactions.json"
    local_paths = glob.glob(local_pattern)
    
    # Combine and deduplicate paths
    all_paths = set(file_paths + local_paths)
    return [Path(p) for p in all_paths]

def sample_interaction_files(n: int) -> List[Path]:
    """Sample n random interaction files from the available files."""
    all_files = find_interaction_files()
    
    if not all_files:
        print("[error] No interaction files found in INTERACTION_CACHE or current directory.", file=sys.stderr)
        sys.exit(1)
        
    if n > len(all_files):
        print(f"[warn] Requested {n} files but only {len(all_files)} are available. Using all files.")
        return all_files
        
    return random.sample(all_files, n)

def extract_gene_pairs_from_filename(filename: str) -> Tuple[str, str]:
    """Extract the gene pair from an interaction filename."""
    # Extract from pattern like network_GENE1_GENE2_interactions.json
    pattern = r'network_([A-Za-z0-9\-]+)_([A-Za-z0-9\-]+)_interactions\.json'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1), match.group(2)
    else:
        # Return empty strings if pattern doesn't match
        return "", ""

def normalize_gene_name(name: str) -> str:
    """
    Normalize gene names to handle special characters and common variations.
    This helps with consistent comparison between different model outputs.
    """
    if not name:
        return name
        
    # Handle specific cases like NF-κB and similar
    replacements = {
        "NF-κB": "NF-KB",
        "NF-kappa-B": "NF-KB",
        "NF-kappaB": "NF-KB",
        "NFκB": "NF-KB",
        "NFkB": "NF-KB",
        "NF-KB": "NF-KB",
        "NF-kB": "NF-KB",
        "p53": "TP53",
        "p21": "CDKN1A",
        "IL-1β": "IL1B",
        "IL-1b": "IL1B",
        "IL1β": "IL1B",
        "TNF-α": "TNFA",
        "TNFα": "TNFA",
        "TNF-a": "TNFA",
        "IκB": "IKB",
        "IkB": "IKB",
        "I-kB": "IKB",
        "TGF-β": "TGFB"
    }
    
    # Direct replacement for known variants
    if name in replacements:
        return replacements[name]
    
    # General normalization
    normalized = name.strip().upper()
    
    # Replace Greek letters with latin equivalents
    normalized = (normalized
        .replace('Α', 'A').replace('α', 'A')
        .replace('Β', 'B').replace('β', 'B')
        .replace('Γ', 'G').replace('γ', 'G')
        .replace('Δ', 'D').replace('δ', 'D')
        .replace('Ε', 'E').replace('ε', 'E')
        .replace('Ζ', 'Z').replace('ζ', 'Z')
        .replace('Η', 'H').replace('η', 'H')
        .replace('Θ', 'TH').replace('θ', 'TH')
        .replace('Ι', 'I').replace('ι', 'I')
        .replace('Κ', 'K').replace('κ', 'K')
        .replace('Λ', 'L').replace('λ', 'L')
        .replace('Μ', 'M').replace('μ', 'M')
        .replace('Ν', 'N').replace('ν', 'N')
        .replace('Ξ', 'X').replace('ξ', 'X')
        .replace('Ο', 'O').replace('ο', 'O')
        .replace('Π', 'P').replace('π', 'P')
        .replace('Ρ', 'R').replace('ρ', 'R')
        .replace('Σ', 'S').replace('σ', 'S').replace('ς', 'S')
        .replace('Τ', 'T').replace('τ', 'T')
        .replace('Υ', 'Y').replace('υ', 'Y')
        .replace('Φ', 'F').replace('φ', 'F')
        .replace('Χ', 'CH').replace('χ', 'CH')
        .replace('Ψ', 'PS').replace('ψ', 'PS')
        .replace('Ω', 'O').replace('ω', 'O')
    )
    
    return normalized

def normalize_mechanism(mechanism: str) -> str:
    """
    Normalize mechanism descriptions to account for minor variations in wording.
    """
    if not mechanism:
        return mechanism
        
    # Strip whitespace and lowercase for consistent comparison
    normalized = mechanism.strip().lower()
    
    # Map synonymous mechanisms to a standard form
    mechanism_map = {
        # Transcriptional regulation
        "transcriptionally activates": "transcriptionally activates",
        "activates transcription of": "transcriptionally activates",
        "activates the transcription of": "transcriptionally activates",
        "induces transcription of": "transcriptionally activates",
        
        # General activation
        "activates": "activates",
        "activation of": "activates",
        "directly activates": "activates",
        
        # Regulation
        "regulates": "regulates",
        "directly regulates": "regulates",
        "regulates the activity of": "regulates",
        
        # Indirect regulation
        "indirectly regulates": "indirectly regulates",
        "indirectly affects": "indirectly regulates",
        "indirect regulation of": "indirectly regulates",
        
        # Interactions
        "interacts with": "interacts with",
        "physically interacts with": "interacts with",
        "forms a complex with": "interacts with",
        "binds to": "interacts with",
        
        # Inhibition
        "inhibits": "inhibits",
        "suppresses": "inhibits",
        "represses": "inhibits",
        "negatively regulates": "inhibits"
    }
    
    # Return the standardized form if it exists, otherwise return the original lowercased form
    for pattern, replacement in mechanism_map.items():
        if normalized == pattern or normalized.startswith(pattern + " "):
            return replacement
            
    return normalized

def normalize_interactions(paths_data: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Normalize interaction data by extracting source-target pairs and their mechanisms.
    Returns a dictionary mapping (source,target) tuples to sets of mechanisms.
    
    Handles special characters in gene names and normalizes them for consistent comparison.
    Also normalizes mechanism descriptions to account for minor variations.
    """
    interactions = defaultdict(set)
    
    # Handle case where there might be an error or no_path response
    if "error" in paths_data or "no_path" in paths_data:
        return interactions
    
    paths = paths_data.get("paths", [])
    if not paths:
        return interactions
        
    for path in paths:
        edges = path.get("edges", [])
        if not edges:
            continue
            
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            mechanism = edge.get("mechanism", "")
            
            if source and target and mechanism:
                # Normalize gene names to handle special characters and variations
                norm_source = normalize_gene_name(source)
                norm_target = normalize_gene_name(target)
                
                # Normalize the mechanism description
                norm_mechanism = normalize_mechanism(mechanism)
                
                interactions[(norm_source, norm_target)].add(norm_mechanism)
    
    return interactions

def gpt_model_analysis(gene_a: str, gene_b: str, model_predictions: Dict[str, Dict[str, Any]], config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Use GPT-4.1 to analyze and compare responses from different models for a gene pair.
    
    Args:
        gene_a: First gene in the pair
        gene_b: Second gene in the pair
        model_predictions: Dictionary of model responses
        
    Returns:
        Analysis results from GPT-4.1
    """
    # Format the model responses for comparison
    model_responses_text = []
    
    for model_name, prediction in model_predictions.items():
        # Skip models with errors
        if "error" in prediction:
            model_responses_text.append(f"## Model: {model_name}\nError: {prediction['error']}")
            continue
            
        # Format paths if available
        if "paths" in prediction and prediction["paths"]:
            paths_text = []
            for i, path in enumerate(prediction["paths"], 1):
                path_text = [f"Path {i} (P={path.get('overall_probability', '?')})"]
                path_text.append(f"Summary: {path.get('summary', 'No summary')}")
                
                # Add edges
                edges = path.get("edges", [])
                if edges:
                    path_text.append("Edges:")
                    for j, edge in enumerate(edges, 1):
                        source = edge.get("source", "?")
                        target = edge.get("target", "?")
                        mechanism = edge.get("mechanism", "?")
                        probability = edge.get("probability", "?")
                        evidence = edge.get("evidence", "No evidence")
                        
                        path_text.append(f"  {j}. {source} → {target} ({mechanism}, P={probability})")
                        path_text.append(f"     Evidence: {evidence}")
                
                paths_text.append("\n".join(path_text))
            
            model_response = f"## Model: {model_name}\n" + "\n\n".join(paths_text)
        elif "no_path" in prediction:
            model_response = f"## Model: {model_name}\nNo path found: {prediction.get('reason', 'No reason provided')}"
        else:
            model_response = f"## Model: {model_name}\nUnexpected response format"
            
        model_responses_text.append(model_response)
    
    # Create the prompt for GPT-4.1
    prompt = f"""
I'm going to show you responses from different language models about potential gene/protein interaction paths between gene {gene_a} and gene {gene_b}.

For each model, I'll show you their predicted interaction paths, including:
- The overall path summary
- The probability assigned to the path
- The specific edges (source → target interactions)
- The mechanisms proposed
- Evidence cited

Please analyze these responses and provide:

1. Consensus Analysis: What is the consensus view on how {gene_a} and {gene_b} interact? Focus on:
   - Are there direct interactions or indirect paths?
   - What mechanisms appear most consistently across models?
   - Which intermediate genes/proteins appear in multiple models' responses?

2. Model Comparison: Which models seem closest to this consensus? Rank the models by how well they align with the consensus view.

3. Open Questions: What questions remain unresolved based on differences between model outputs? Identify:
   - Conflicting mechanisms proposed by different models
   - Significantly different probability estimates for the same interactions
   - Different intermediate nodes/paths proposed
   - Contradictory evidence citations

4. Reliability Assessment: Based on the evidence cited, which paths or interactions seem most credible?

Here are the model responses:

{chr(10).join(model_responses_text)}
"""

    # Query GPT-4.1 for analysis
    messages = [
        {"role": "system", "content": "You are an expert systems biologist analyzing different model predictions about gene interactions."},
        {"role": "user", "content": prompt}
    ]
    
    print(f"[info] Sending {gene_a}-{gene_b} model responses to GPT-4.1 for comparative analysis...")
    analysis_response = gpt_call(messages, model="gpt-4.1", config_file=config_file)
    
    # Format the response into a dictionary
    analysis = {
        "gene_a": gene_a,
        "gene_b": gene_b,
        "comparative_analysis": analysis_response,
        "raw_model_responses": model_predictions
    }
    
    return analysis

def compare_predictions(original_data: Dict[str, Any], model_predictions: Dict[str, Dict[str, Any]], config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare the original data with predictions from multiple models.
    First performs traditional comparison, then uses GPT-4.1 for in-depth analysis.
    """
    # Extract gene names from the model predictions or original data
    gene_a = ""
    gene_b = ""
    
    # Try to extract from paths in model predictions first
    for model_name, pred in model_predictions.items():
        if "paths" in pred and pred["paths"] and len(pred["paths"]) > 0:
            path = pred["paths"][0]
            if "edges" in path and path["edges"] and len(path["edges"]) > 0:
                edge = path["edges"][0]
                gene_a = edge.get("source", "")
                if len(path["edges"]) > 1:
                    gene_b = path["edges"][-1].get("target", "")
                else:
                    gene_b = edge.get("target", "")
                break
    
    # If still not found, try original data
    if not gene_a or not gene_b:
        if "paths" in original_data and original_data["paths"] and len(original_data["paths"]) > 0:
            path = original_data["paths"][0]
            if "edges" in path and path["edges"] and len(path["edges"]) > 0:
                edge = path["edges"][0]
                gene_a = edge.get("source", "")
                if len(path["edges"]) > 1:
                    gene_b = path["edges"][-1].get("target", "")
                else:
                    gene_b = edge.get("target", "")
    
    # Normalize the original interactions
    original_interactions = normalize_interactions(original_data)
    
    # Normalize the model predictions
    model_interactions = {}
    for model_name, prediction in model_predictions.items():
        model_interactions[model_name] = normalize_interactions(prediction)
    
    # Initialize comparison results
    comparison = {
        "agreement": {
            "full": [],
            "partial": []
        },
        "disagreement": {
            "mechanisms": [],
            "connections": []
        },
        "unique_predictions": {}
    }
    
    # Track mechanisms for each source-target pair across all models
    all_source_target_pairs = set()
    all_source_target_pairs.update(original_interactions.keys())
    for model_name, interactions in model_interactions.items():
        all_source_target_pairs.update(interactions.keys())
        comparison["unique_predictions"][model_name] = []
    
    # Compare interactions across all models
    for source, target in all_source_target_pairs:
        # Get mechanisms from original data
        original_mechanisms = original_interactions.get((source, target), set())
        
        # Track which models predict this pair and with what mechanisms
        model_predictions_for_pair = {}
        for model_name, interactions in model_interactions.items():
            predicted_mechanisms = interactions.get((source, target), set())
            if predicted_mechanisms:
                model_predictions_for_pair[model_name] = predicted_mechanisms
        
        # Check for agreement/disagreement
        if len(model_predictions_for_pair) == 0:
            # Only in original data, not predicted by any model
            continue
            
        if (source, target) in original_interactions:
            # This pair exists in original data
            agreement_count = 0
            partial_agreement_count = 0
            disagreement_count = 0
            
            for model_name, mechanisms in model_predictions_for_pair.items():
                if mechanisms == original_mechanisms:
                    agreement_count += 1
                elif mechanisms.intersection(original_mechanisms):
                    partial_agreement_count += 1
                else:
                    disagreement_count += 1
            
            # Determine overall agreement status
            if agreement_count == len(model_predictions_for_pair):
                # All models fully agree with original
                comparison["agreement"]["full"].append({
                    "source": source,
                    "target": target,
                    "mechanisms": list(original_mechanisms),
                    "models": list(model_predictions_for_pair.keys())
                })
            elif partial_agreement_count + agreement_count == len(model_predictions_for_pair):
                # All models at least partially agree with original
                comparison["agreement"]["partial"].append({
                    "source": source,
                    "target": target,
                    "original_mechanisms": list(original_mechanisms),
                    "model_predictions": {model: list(mechs) for model, mechs in model_predictions_for_pair.items()}
                })
            else:
                # Some models disagree completely
                comparison["disagreement"]["mechanisms"].append({
                    "source": source,
                    "target": target,
                    "original_mechanisms": list(original_mechanisms),
                    "model_predictions": {model: list(mechs) for model, mechs in model_predictions_for_pair.items()}
                })
        else:
            # This pair is not in original data - it's a unique prediction
            for model_name, mechanisms in model_predictions_for_pair.items():
                comparison["unique_predictions"][model_name].append({
                    "source": source,
                    "target": target,
                    "mechanisms": list(mechanisms)
                })
    
    # Check for connections in original data that no model predicted
    for source, target in original_interactions.keys():
        if all((source, target) not in model_ints for model_ints in model_interactions.values()):
            comparison["disagreement"]["connections"].append({
                "source": source,
                "target": target,
                "original_mechanisms": list(original_interactions[(source, target)]),
                "note": "No model predicted this connection"
            })
    
    # Get GPT-4.1 comparative analysis if we have gene names
    if gene_a and gene_b:
        gpt_analysis = gpt_model_analysis(gene_a, gene_b, model_predictions, config_file)
        comparison["gpt_comparative_analysis"] = gpt_analysis
    
    return comparison

def print_console_summary(gene_a: str, gene_b: str, comparison: Dict[str, Any], models: List[str]) -> None:
    """
    Print a concise summary of model agreement and disagreement for a gene pair.
    This provides immediate feedback on the console during analysis.
    """
    # Check if we have a GPT-4.1 comparative analysis
    has_gpt_analysis = "gpt_comparative_analysis" in comparison and comparison["gpt_comparative_analysis"] is not None
    
    # Print header with gene pair info
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR GENE PAIR: {gene_a} - {gene_b}")
    print(f"{'='*80}")
    
    # Print traditional model agreement summary (brief version)
    # Get counts for different categories
    full_agreement_count = len(comparison["agreement"]["full"])
    partial_agreement_count = len(comparison["agreement"]["partial"])
    disagreement_mechanisms_count = len(comparison["disagreement"]["mechanisms"])
    disagreement_connections_count = len(comparison["disagreement"]["connections"])
    
    unique_predictions = {
        model: len(comparison["unique_predictions"].get(model, [])) 
        for model in models
    }
    
    # Calculate total interaction count
    total_interactions = full_agreement_count + partial_agreement_count + disagreement_mechanisms_count + disagreement_connections_count
    
    # Display basic agreement stats
    if total_interactions == 0:
        print("[Traditional Analysis] No common interactions found between models")
    else:
        # Calculate agreement percentages
        agreement_percent = (full_agreement_count / total_interactions) * 100 if total_interactions > 0 else 0
        partial_percent = (partial_agreement_count / total_interactions) * 100 if total_interactions > 0 else 0
        
        print("\n[Traditional Analysis Summary]")
        print(f"  - Full agreement: {full_agreement_count}/{total_interactions} interactions ({agreement_percent:.1f}%)")
        print(f"  - Partial agreement: {partial_agreement_count}/{total_interactions} interactions ({partial_percent:.1f}%)")
        print(f"  - Disagreements: {disagreement_mechanisms_count + disagreement_connections_count}/{total_interactions} interactions")
        
    # If we have GPT-4.1 analysis, print it in a nicely formatted way
    if has_gpt_analysis:
        analysis = comparison["gpt_comparative_analysis"]["comparative_analysis"]
        
        print(f"\n{'='*80}")
        print(f"GPT-4.1 COMPARATIVE ANALYSIS")
        print(f"{'='*80}")
        
        # Format and print the analysis
        # Split the analysis into sections for better readability
        formatted_analysis = analysis
        
        # Check if the analysis has section headers (numbers followed by dot and colon)
        for section in ["1.", "2.", "3.", "4."]:
            # Highlight section headers
            formatted_analysis = formatted_analysis.replace(f"{section}", f"\n{section}")
        
        # Print the formatted analysis
        print(formatted_analysis)
        
        print(f"\n{'='*80}")
    else:
        print("\n[Note] No GPT-4.1 comparative analysis available for this gene pair.")
        
    # Add a final separator
    print(f"{'='*80}\n")

def generate_new_predictions(gene_a: str, gene_b: str, models: List[Dict[str, str]], n_paths: int = 3, 
                            max_retries: int = 2, config_file: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Generate new predictions for a gene pair using multiple models."""
    predictions = {}
    
    for model_info in models:
        model_name = model_info["shortname"]
        model_id = model_info["model"]
        
        print(f"[info] Generating predictions for {gene_a}-{gene_b} using model {model_name} ({model_id})...")
        
        # Try up to max_retries times
        for attempt in range(max_retries):
            try:
                result = query_paths(gene_a, gene_b, n_paths, model_id, config_file)
                predictions[model_name] = result
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[warn] Error with model {model_name} (attempt {attempt+1}/{max_retries}): {e}")
                    time.sleep(2)  # Short delay before retry
                else:
                    print(f"[error] Failed to get predictions from model {model_name} after {max_retries} attempts: {e}")
                    predictions[model_name] = {"error": str(e)}
    
    return predictions

def analyze_file(file_path: Path, models: List[Dict[str, str]], n_paths: int = 3, config_file: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a single interaction file with multiple models."""
    try:
        # Parse the interaction file
        with open(file_path, 'r') as f:
            original_data = json.load(f)
        
        # Extract gene pair from filename
        gene_a, gene_b = extract_gene_pairs_from_filename(file_path.name)
        if not gene_a or not gene_b:
            print(f"[warn] Could not extract gene pair from filename: {file_path.name}")
            return {"error": f"Could not extract gene pair from filename: {file_path.name}"}
        
        # Generate predictions from all models
        model_predictions = generate_new_predictions(gene_a, gene_b, models, n_paths, config_file=config_file)
        
        # Compare the predictions
        comparison = compare_predictions(original_data, model_predictions, config_file)
        
        # Print a detailed console summary of the results, including GPT-4.1 analysis
        print_console_summary(gene_a, gene_b, comparison, [m["shortname"] for m in models])
        
        # Add metadata
        result = {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "file_path": str(file_path),
            "comparison": comparison,
            "original_data": original_data,
            "model_predictions": model_predictions
        }
        
        return result
    
    except Exception as e:
        print(f"[error] Error analyzing file {file_path}: {e}")
        traceback.print_exc()
        return {"error": str(e), "file_path": str(file_path)}

def main() -> None:
    """Main function to run the comparison of model predictions."""
    parser = argparse.ArgumentParser(
        description="Compare interaction predictions from different models on gene pairs"
    )
    parser.add_argument(
        "--pairs", type=int, default=10,
        help="Number of random gene pairs to analyze (default: 10)"
    )
    parser.add_argument(
        "--paths", type=int, default=3,
        help="Number of paths to request from each model (default: 3)"
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Specific model shortnames to use (default: all available models)"
    )
    parser.add_argument(
        "--out", default="model_comparison_results.json",
        help="Output file for comparison results (default: model_comparison_results.json)"
    )
    parser.add_argument(
        "--skip-api-calls", action="store_true",
        help="Skip making API calls and just analyze existing files"
    )
    parser.add_argument(
        "--seed", type=int, 
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--timeout", type=int, default=10,
        help="Timeout in seconds for endpoint availability test (default: 10)"
    )
    parser.add_argument(
        "--skip-endpoint-check", action="store_true",
        help="Skip checking if model endpoints are available"
    )
    parser.add_argument(
        "--config", dest="config_file",
        help="Path to model configuration file (default: model_servers.yaml)"
    )
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Ensure API key unless skipping API calls
    if not args.skip_api_calls and not os.getenv("OPENAI_API_KEY") and not MODEL_CONFIG_AVAILABLE:
        print("[error] OPENAI_API_KEY environment variable is not set and model_config is not available.", file=sys.stderr)
        sys.exit(1)
    
    # Get all available models from configuration
    all_models = get_available_models(args.config_file)
    
    # Filter to requested models if specified
    if args.models:
        selected_models = [m for m in all_models if m["shortname"] in args.models]
        if not selected_models:
            print(f"[error] None of the requested models {args.models} were found.", file=sys.stderr)
            print(f"[info] Available models: {[m['shortname'] for m in all_models]}")
            sys.exit(1)
        potential_models = selected_models
    else:
        potential_models = all_models
    
    # Test which model endpoints are available
    if not args.skip_api_calls and not args.skip_endpoint_check:
        print("[info] Testing model endpoints availability...")
        
        # Test each model endpoint
        available_models = []
        unavailable_models = []
        
        for model in potential_models:
            if test_model_endpoint(model, timeout=args.timeout, config_file=args.config_file):
                available_models.append(model)
            else:
                unavailable_models.append(model)
        
        if not available_models:
            print("[error] No model endpoints are available. Please check your network connection and API keys.", file=sys.stderr)
            sys.exit(1)
        
        if unavailable_models:
            print(f"[warn] Skipping {len(unavailable_models)} unavailable model(s): {[m['shortname'] for m in unavailable_models]}")
        
        models = available_models
    else:
        # Skip endpoint checking
        models = potential_models
    
    print(f"[info] Using {len(models)} model(s): {[m['shortname'] for m in models]}")
    
    # Sample interaction files
    sample_files = sample_interaction_files(args.pairs)
    print(f"[info] Sampled {len(sample_files)} interaction files for analysis")
    
    # Analyze each file
    results = []
    for i, file_path in enumerate(sample_files, 1):
        print(f"\n[info] Analyzing file {i}/{len(sample_files)}: {file_path.name}")
        
        if args.skip_api_calls:
            # Just record the file path without making API calls
            results.append({
                "file_path": str(file_path),
                "gene_a": extract_gene_pairs_from_filename(file_path.name)[0],
                "gene_b": extract_gene_pairs_from_filename(file_path.name)[1],
                "skipped": True
            })
        else:
            result = analyze_file(file_path, models, args.paths, args.config_file)
            results.append(result)
            
            # Save intermediate results after each file
            with open(f"{args.out}.partial", 'w') as f:
                json.dump({
                    "completed": i,
                    "total": len(sample_files),
                    "models": [m["shortname"] for m in models],
                    "results": results
                }, f, indent=2)
    
    # Calculate summary statistics
    summary = {
        "total_files_analyzed": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "full_agreement_count": sum(len(r.get("comparison", {}).get("agreement", {}).get("full", [])) 
                                 for r in results if "comparison" in r),
        "partial_agreement_count": sum(len(r.get("comparison", {}).get("agreement", {}).get("partial", [])) 
                                    for r in results if "comparison" in r),
        "disagreement_count": sum(len(r.get("comparison", {}).get("disagreement", {}).get("mechanisms", [])) 
                                + len(r.get("comparison", {}).get("disagreement", {}).get("connections", [])) 
                                for r in results if "comparison" in r),
        "unique_predictions_by_model": {
            model["shortname"]: sum(len(r.get("comparison", {}).get("unique_predictions", {}).get(model["shortname"], [])) 
                                  for r in results if "comparison" in r)
            for model in models
        }
    }
    
    # Save the final results
    with open(args.out, 'w') as f:
        json.dump({
            "summary": summary,
            "models": [m["shortname"] for m in models],
            "results": results
        }, f, indent=2)
    
    print(f"[info] Saved comparison results to {args.out}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total files analyzed: {summary['total_files_analyzed']}")
    print(f"Files with errors: {summary['errors']}")
    print(f"Full agreement count: {summary['full_agreement_count']}")
    print(f"Partial agreement count: {summary['partial_agreement_count']}")
    print(f"Disagreement count: {summary['disagreement_count']}")
    print("Unique predictions by model:")
    for model, count in summary["unique_predictions_by_model"].items():
        print(f"  - {model}: {count}")

if __name__ == "__main__":
    main()