#!/usr/bin/env python3
"""
Extract gene/protein entities from context, generate pairs.txt, and invoke gene_chain_v1.py.

Skips generating interaction data for pairs that have already been processed (checks for
existing network_[GeneA]_[GeneB]_interactions.json files in both the current directory
and the INTERACTION_CACHE directory).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

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


def gpt_call(messages: List[Dict[str, str]], *, model: str = "gpt-4.1",
             max_tokens: int = 2048, temperature: float = 0.2,
             retries: int = 3, backoff: float = 5.0) -> str:
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
            print(f"[warn] OpenAI call failed on attempt {attempt}: {exc}. Retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("gpt_call exhausted retries without success")

def extract_entities(context: str, model: str) -> List[str]:
    """Extract gene and protein names from unstructured context text."""
    sys_prompt = (
        "You are a knowledgeable text-mining engine for biology. "
        "When given unstructured text, you extract all gene and protein names mentioned."
    )
    user_prompt = (
        "Extract all genes and proteins mentioned in the following text. "
        "Return a JSON object with key 'entities' listing unique names. "
        "Only output the JSON object without markdown or commentary.\n\n"
        + context
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = gpt_call(messages, model=model)
    try:
        # First try standard JSON parsing
        obj = json.loads(raw)
        return list({str(e).strip() for e in obj.get("entities", []) if e})
    except json.JSONDecodeError:
        # If standard parsing fails, try cleaning and fixing the raw output
        try:
            # Try to clean up common JSON issues
            # 1. Remove any markdown code block markers
            cleaned_raw = re.sub(r'```(?:json)?\s*|\s*```', '', raw)
            
            # 2. Attempt to fix unquoted or single-quoted properties
            # Replace single quotes with double quotes, but be careful with already proper double quotes
            cleaned_raw = re.sub(r'(?<!")\'(?!")', '"', cleaned_raw)
            
            # Try parsing again with the cleaned version
            obj = json.loads(cleaned_raw)
            return list({str(e).strip() for e in obj.get("entities", []) if e})
        except Exception:
            # If all attempts fail, raise a clearer error
            print(f"[error] Failed to parse entities JSON. Raw output:\n{raw}", file=sys.stderr)
            raise

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract entities from context and generate pairs for gene_chain_v1.py. Skips pairs with existing analysis."
    )
    parser.add_argument("-i", "--input-file", required=True,
                        help="Path to text file with unstructured biological context")
    parser.add_argument("--model", default="gpt-4.1",
                        help="OpenAI model name (default: gpt-4.1)")
    parser.add_argument("--pairs-file", default="pairs.txt",
                        help="Path to output pairs file (default: pairs.txt)")
    parser.add_argument("--script", default="gene_chain_v1.py",
                        help="Path to gene_chain_v1.py script to invoke")
    parser.add_argument("--model-for-chain", default="gpt-4.1",
                        help="OpenAI model for gene_chain_v1.py (default: gpt-4.1)")
    parser.add_argument("--paths", type=int, default=3,
                        help="Maximum number of paths to request in gene_chain_v1.py (default: 3)")
    parser.add_argument("--out", default="network",
                        help="Output file prefix for gene_chain_v1.py (default: network)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        context = Path(args.input_file).read_text()
    except Exception as exc:
        print(f"[error] Could not read input file: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[info] Extracting entities from context...")
    entities = extract_entities(context, args.model)
    if not entities:
        print("[warn] No entities extracted.", file=sys.stderr)

        # Check for existing interaction files
    def get_existing_pairs() -> Set[Tuple[str, str]]:
        """Find all pairs that already have interaction data in current dir and cache."""
        existing_pairs = set()
        
        # Search patterns to check
        patterns = [
            "network_*_*_interactions.json",  # Standard pattern
            "*_*_interactions.json",          # Any prefix with two genes
        ]
        
        # Search locations
        search_dirs = [".", str(INTERACTION_CACHE)]
        
        for search_dir in search_dirs:
            for pattern in patterns:
                search_pattern = os.path.join(search_dir, pattern)
                for filename in glob.glob(search_pattern):
                    # Extract the gene names from the filename
                    base_filename = os.path.basename(filename)
                    
                    if base_filename.startswith("network_"):
                        parts = base_filename.replace('network_', '').replace('_interactions.json', '').split('_')
                    else:
                        parts = base_filename.replace('_interactions.json', '').split('_')
                        
                    if len(parts) >= 2:
                        # Add both orderings to ensure we catch all variations
                        gene_a, gene_b = parts[0], parts[1]
                        existing_pairs.add((gene_a, gene_b))
                        existing_pairs.add((gene_b, gene_a))
        
        return existing_pairs
    
    existing_pairs = get_existing_pairs()
    print(f"[info] Found {len(existing_pairs)//2} existing interaction pairs")

    # Generate all unique unordered pairs, filtering out existing ones
    pairs = []
    skipped_pairs = []
    
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            gene_a, gene_b = entities[i], entities[j]
            pair = (gene_a, gene_b)
            
            # Check both orderings
            if pair in existing_pairs or (gene_b, gene_a) in existing_pairs:
                # Skip this pair as it's already been processed
                skipped_pairs.append(pair)
                print(f"[info] Skipping existing pair: {gene_a} - {gene_b}")
            else:
                pairs.append(pair)

    if not pairs:
        print("[info] All potential pairs already processed. Nothing to do.")
        if skipped_pairs:
            print(f"[info] Skipped {len(skipped_pairs)} existing pairs.")
        sys.exit(0)

    # Write new pairs to file
    pairs_path = Path(args.pairs_file)
    try:
        with pairs_path.open("w") as f:
            for a, b in pairs:
                f.write(f"{a} {b}\n")
    except Exception as exc:
        print(f"[error] Could not write pairs file {pairs_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Wrote {len(pairs)} new pairs to {pairs_path}")
    if skipped_pairs:
        print(f"[info] Skipped {len(skipped_pairs)} existing pairs")

    # Check if all pairs already have output files
    if not pairs:
        print("[info] All pairs already have existing output files. Skipping gene_chain_v1.py invocation.")
        sys.exit(0)
        
    # Invoke gene_chain_v1.py on the pairs file, only if we have pairs to process
    cmd = [
        sys.executable, 
        args.script, 
        "-i", str(pairs_path),
        "--model", args.model_for_chain,
        "--paths", str(args.paths),
        "--out", args.out
    ]
    print(f"[info] Invoking gene_chain_v1.py: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Report the outcome
    if result.returncode == 0:
        print(f"[info] Successfully processed {len(pairs)} new gene pairs")
        print(f"[info] Skipped {len(skipped_pairs)} existing pairs")
    else:
        print(f"[error] gene_chain_v1.py failed with return code {result.returncode}")
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
