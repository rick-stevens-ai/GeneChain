#!/usr/bin/env python3
"""
Interactions to Biological Summary
===================================

This script reads a gene interaction JSON file (like those produced by gene_chain_v1.py)
and uses OpenAI's GPT-4.1 to generate a comprehensive biological summary of the 
interactions, providing context, mechanisms, and related biomolecules.

Usage:
------
$ export OPENAI_API_KEY="sk-..."
$ python interactions_to_summary.py --input-file network_GADD45A_ZMAT3_interactions.json --output-file summary.txt

Requirements:
------------
- Python ≥3.8
- openai
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Import model configuration
try:
    from model_config import get_model_config
    MODEL_CONFIG_AVAILABLE = True
except ImportError:
    MODEL_CONFIG_AVAILABLE = False
    print("[warn] model_config module not found; using default OpenAI configuration")

# Define the cache directory for interaction data
INTERACTION_CACHE = Path("INTERACTION_CACHE")
if not INTERACTION_CACHE.exists():
    INTERACTION_CACHE.mkdir(exist_ok=True)


def gpt_call(messages: List[Dict[str, str]], *, 
             model: str = "gpt-4.1", 
             max_tokens: int = 2048,
             temperature: float = 0.2, 
             retries: int = 3, 
             backoff: float = 5.0) -> str:
    """Robust wrapper around openai.ChatCompletion.create with simple retry."""
    import openai
    global MODEL_CONFIG_AVAILABLE
    
    # Configure OpenAI client based on model_config if available
    if MODEL_CONFIG_AVAILABLE:
        try:
            # Try to get model configuration by name or shortname
            model_config = None
            try:
                # First try by model name
                model_config = get_model_config(model)
            except ValueError:
                # Then try by shortname
                try:
                    model_config = get_model_config(model, by_shortname=True)
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
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message["content"].strip()
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"[warn] OpenAI call failed on attempt {attempt}: {exc}. Retrying in {backoff} s…",
                  file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("Unreachable code in gpt_call")


def find_interaction_file(input_file: str) -> Optional[Path]:
    """
    Find the interaction file in either the current directory or the cache.
    Accepts either a direct path or a base filename.
    """
    # First check if the input_file is a direct path that exists
    path = Path(input_file)
    if path.exists():
        return path
    
    # Check if the file exists in the INTERACTION_CACHE
    cache_path = INTERACTION_CACHE / path.name
    if cache_path.exists():
        return cache_path
    
    # If only a filename like "file.json" was provided, check both locations
    if path.name == input_file:
        local_path = Path(input_file)
        if local_path.exists():
            return local_path
        
        cache_path = INTERACTION_CACHE / input_file
        if cache_path.exists():
            return cache_path
    
    return None

def extract_gene_names_from_filename(filename: str) -> Tuple[str, str]:
    """Extract gene names from the interaction filename."""
    # Try to extract gene names from pattern like network_GENE1_GENE2_interactions.json
    pattern = r'(?:network_)?([A-Za-z0-9]+)_([A-Za-z0-9]+)(?:_interactions)?\.json'
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1), match.group(2)
    else:
        # Return empty strings if pattern doesn't match
        return "", ""


def generate_summary(interactions_data: Dict[str, Any], 
                     gene1: str, 
                     gene2: str,
                     model: str = "gpt-4.1",
                     max_tokens: int = 1000) -> str:
    """Generate a biological summary from the interactions data using GPT-4."""
    # Create a concise representation of the interaction data
    paths_data = interactions_data.get("paths", [])
    
    # Extract key information from the paths
    paths_summary = []
    for i, path in enumerate(paths_data, 1):
        edges = path.get("edges", [])
        path_prob = path.get("overall_probability", "N/A")
        path_desc = path.get("summary", "No summary available")
        
        edges_summary = []
        for edge in edges:
            src = edge.get("source", "Unknown")
            tgt = edge.get("target", "Unknown")
            mech = edge.get("mechanism", "Unknown")
            prob = edge.get("probability", "N/A")
            evid = edge.get("evidence", "")
            
            edges_summary.append(f"{src} {mech} {tgt} (P={prob}) [Evidence: {evid}]")
        
        paths_summary.append({
            "path_number": i,
            "probability": path_prob,
            "summary": path_desc,
            "edges": edges_summary
        })
    
    # Create a comprehensive prompt for GPT-4
    system_prompt = (
        "You are a computational biologist summarizing interaction data between genes/proteins. "
        "Your summaries provide comprehensive biological context while remaining concise and on-topic."
    )
    
    user_prompt = (
        f"Please summarize the interactions between {gene1} and {gene2} based on the following data:\n\n"
        "INTERACTION PATHS:\n"
    )
    
    # Add paths information
    for path in paths_summary:
        user_prompt += f"\nPath {path['path_number']} (P={path['probability']}):\n"
        user_prompt += f"Summary: {path['summary']}\n"
        user_prompt += "Edges:\n"
        for edge in path['edges']:
            user_prompt += f"- {edge}\n"
    
    user_prompt += (
        "\nProvide a comprehensive biological summary (no more than 1000 tokens) that includes:\n"
        "1. Brief descriptions of the roles and functions of the primary genes/proteins\n"
        "2. The nature and significance of the interactions\n"
        "3. The biological conditions or contexts where these interactions occur\n"
        "4. Other key genes, proteins, or biomolecules that are involved\n"
        "5. Potential biological implications or significance\n\n"
        "Prioritize accuracy, concision, and clinical/biological relevance."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Call the model with a smaller token limit
    return gpt_call(messages, model=model, max_tokens=max_tokens, temperature=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate biological summaries from gene interaction data using GPT-4"
    )
    parser.add_argument(
        "--input-file", required=True,
        help="Path to the interactions JSON file (e.g., network_GENE1_GENE2_interactions.json)"
    )
    parser.add_argument(
        "--output-file", 
        help="Path to save the summary (defaults to summary_GENE1_GENE2.txt)"
    )
    parser.add_argument(
        "--model", default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000,
        help="Maximum tokens for the summary (default: 1000)"
    )
    parser.add_argument(
        "--cache-output", action="store_true",
        help="Also save the summary to the INTERACTION_CACHE directory"
    )
    args = parser.parse_args()
    
    # Ensure API key
    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    
    # Find the interaction file in either current directory or cache
    interaction_file = find_interaction_file(args.input_file)
    if not interaction_file:
        print(f"[error] Interaction file not found: {args.input_file}", file=sys.stderr)
        print(f"[info] Searched in current directory and {INTERACTION_CACHE}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[info] Using interaction file: {interaction_file}")
    
    # Parse input file
    try:
        with open(interaction_file, 'r') as f:
            interactions_data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to load interactions file {interaction_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract gene names from the filename
    gene1, gene2 = extract_gene_names_from_filename(os.path.basename(str(interaction_file)))
    
    # If genes couldn't be extracted from filename, try to infer from data
    if not gene1 or not gene2:
        print("[warn] Could not extract gene names from filename - attempting to infer from data...", file=sys.stderr)
        try:
            paths = interactions_data.get("paths", [])
            if paths and "edges" in paths[0] and len(paths[0]["edges"]) > 0:
                # Get the first and last nodes in the first path
                gene1 = paths[0]["edges"][0]["source"]
                gene2 = paths[0]["edges"][-1]["target"]
                print(f"[info] Inferred gene names: {gene1} and {gene2}", file=sys.stderr)
            else:
                print("[warn] Could not infer gene names from data - using placeholders", file=sys.stderr)
                gene1, gene2 = "Gene1", "Gene2"
        except Exception:
            print("[warn] Error inferring gene names - using placeholders", file=sys.stderr)
            gene1, gene2 = "Gene1", "Gene2"
    
    # Generate summary
    print(f"[info] Generating biological summary for {gene1}-{gene2} interactions...", file=sys.stderr)
    try:
        summary = generate_summary(
            interactions_data, 
            gene1, 
            gene2, 
            model=args.model,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        print(f"[error] Failed to generate summary: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"summary_{gene1}_{gene2}.txt"
    
    output_files = []
    
    # Add the local output file
    output_files.append(Path(output_file))
    
    # Add cache output file if requested
    if args.cache_output:
        cache_output = INTERACTION_CACHE / f"summary_{gene1}_{gene2}.txt"
        output_files.append(cache_output)
    
    # Write to output files
    for out_file in output_files:
        try:
            with open(out_file, 'w') as f:
                f.write(f"# Biological Summary: {gene1} and {gene2} Interactions\n\n")
                f.write(summary)
            print(f"[info] Saved biological summary to {out_file}", file=sys.stderr)
        except Exception as e:
            print(f"[error] Failed to write summary to {out_file}: {e}", file=sys.stderr)
    
    if not output_files:
        print("\n" + summary)  # Output to console as fallback
    
    # Print confirmation with character count info
    char_count = len(summary)
    approx_tokens = char_count // 4  # Rough approximation of GPT tokens
    print(f"[info] Generated summary with ~{char_count} characters (~{approx_tokens} tokens)")


if __name__ == "__main__":
    main()