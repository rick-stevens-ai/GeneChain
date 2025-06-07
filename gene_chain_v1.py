#!/usr/bin/env python3
"""
Gene/Protein Interaction Chain Finder
=====================================

This script queries the OpenAI API (GPT‑4) to discover mechanistic interaction
chains that connect two genes or proteins.  Each interaction in the chain is
annotated with a probability, and multiple paths (shortest plus informative
longer alternatives) are returned.  The script then builds a network diagram
from the returned paths and writes:

1. `interactions.json` – the raw structured response from the model.
2. `network.dot`          – a Graphviz DOT file of the interaction network.
3. `network.png`          – (optional) a rendered PNG if Graphviz is available
                            or if pydot/Graphviz are installed.

Usage
-----
$ export OPENAI_API_KEY="sk‑..."
$ python gene_chain_v1.py TP53 EGFR --model gpt-4o --paths 4

Requirements
------------
- Python ≥3.8
- openai           (``pip install openai``)
- networkx         (``pip install networkx``)
- matplotlib       (for PNG rendering; ``pip install matplotlib``)
- pydot & Graphviz (optional but recommended for high‑quality PNGs)

Notes
-----
* The model output is *hypothesis‑level* knowledge; validate paths against
  primary literature or databases before drawing strong conclusions.
* Large context prompts may incur non‑trivial token costs.
* The chain probabilities are subjective estimates from the language model.
* The script checks for existing interaction files before making API calls
  to avoid redundant processing and unnecessary API costs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

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

# Function to safely resolve file paths, preventing double INTERACTION_CACHE issue
def get_safe_path(base: str, suffix: str, use_cache: bool = True) -> Path:
    """
    Create a safe path that avoids the INTERACTION_CACHE/INTERACTION_CACHE issue.
    If base already contains INTERACTION_CACHE, it strips it out before creating the path.
    """
    # If base already includes INTERACTION_CACHE, remove it
    if isinstance(base, str) and "INTERACTION_CACHE" in base:
        base = base.replace("INTERACTION_CACHE/", "").replace("INTERACTION_CACHE", "")
        # Remove any leading slashes
        base = base.lstrip("/")
    
    # Return either the cache path or the local path
    if use_cache:
        return INTERACTION_CACHE / f"{base}{suffix}"
    else:
        return Path(f"{base}{suffix}")

import openai
import networkx as nx
import re

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# OpenAI call helpers
# ---------------------------------------------------------------------------

def gpt_call(messages: List[Dict[str, str]], *, model: str = "gpt-4.1",
             temperature: float = 0.2, retries: int = 3, backoff: float = 5.0) -> str:
    """Robust wrapper around openai.ChatCompletion.create with simple retry."""
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
            # For o3 and o4-mini models, don't include temperature parameter
            if "o3" in model or "o4-mini" in model:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            return response.choices[0].message["content"].strip()
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"[warn] OpenAI call failed on attempt {attempt}: {exc}.  Retrying in {backoff} s…",
                  file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("Unreachable code in gpt_call")


def query_paths(entity_a: str, entity_b: str, n_paths: int, model: str) -> Dict[str, Any]:
    """Ask GPT for interaction paths between *entity_a* and *entity_b*."""
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
        "Do *not* wrap the JSON in markdown fences.  Output nothing except the JSON object."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw = gpt_call(messages, model=model)
    try:
        # First try standard JSON parsing
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # If standard parsing fails, try cleaning and fixing the raw output
        try:
            # Try to clean up common JSON issues
            # 1. Remove any markdown code block markers
            cleaned_raw = re.sub(r'```(?:json)?\s*|\s*```', '', raw)
            
            # 2. Attempt to fix unquoted or single-quoted properties
            # Replace single quotes with double quotes, but be careful with already proper double quotes
            cleaned_raw = re.sub(r'(?<!")\'(?!")', '"', cleaned_raw)
            
            # Try parsing again with the cleaned version
            return json.loads(cleaned_raw)
        except Exception:
            # If all attempts fail, raise the original error with raw output for debugging
            raise ValueError(f"Model returned invalid JSON. Raw output was:\n{raw}\n") from exc

# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_graph(paths: List[Dict[str, Any]]) -> nx.MultiDiGraph:
    """Convert list of path dicts into a MultiDiGraph."""
    G = nx.MultiDiGraph()
    for path in paths:
        for edge in path["edges"]:
            src = edge["source"]
            tgt = edge["target"]
            
            # Handle mechanism field - normalize underscore-separated words
            raw_mech = edge["mechanism"]
            # Process the mechanism string: replace underscores with spaces but handle special cases
            if isinstance(raw_mech, str):
                # Handle the case where mechanism has underscores (like "interacts_with")
                # but preserve parenthetical content if present
                parts = raw_mech.split(" (", 1)
                if len(parts) > 1:
                    # Has parenthetical content
                    base_mech = parts[0].replace("_", " ")
                    parenthetical = parts[1]
                    mech = f"{base_mech} ({parenthetical}"
                else:
                    # No parenthetical content
                    mech = raw_mech.replace("_", " ")
            else:
                # Not a string, keep as is (shouldn't happen, but safer)
                mech = raw_mech
                
            prob = float(edge["probability"])
            evid = edge.get("evidence", "?")
            G.add_edge(src, tgt, label=mech, probability=prob, evidence=evid)
    return G


def dump_graphviz(G: nx.MultiDiGraph, filepath: Path):
    """Write the graph to *filepath* in Graphviz dot format."""
    # Create a copy of the graph to avoid modifying the original
    G_copy = G.copy()
    
    # Ensure all edge labels are properly formatted for DOT output
    for u, v, k, d in G_copy.edges(data=True, keys=True):
        # Ensure label is a string
        if 'label' in d and d['label'] is not None:
            # Make sure the label is properly escaped for DOT format
            # This helps with labels containing special characters or formatting
            label_str = str(d['label'])
            # Escape quotes if needed
            label_str = label_str.replace('"', '\\"')
            G_copy[u][v][k]['label'] = label_str
    
    nx.drawing.nx_pydot.write_dot(G_copy, str(filepath))


def draw_graph_png(G: nx.MultiDiGraph, filepath: Path):
    """Render a quick PNG using matplotlib (falls back if matplotlib missing)."""
    if plt is None:
        print("[info] matplotlib not available – skipping PNG render.")
        return

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    # Node + edge styling
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=9)
    # Consolidate parallel edges into a single label per pair for clarity
    simple_edges = {(u, v): [] for u, v in G.edges()}
    for u, v, d in G.edges(data=True):
        # Process the edge label to handle underscore formatting consistently
        label = d.get("label", "")
        if isinstance(label, str):
            # Split the label to handle parenthetical content properly
            parts = label.split(" (", 1)
            if len(parts) > 1:
                # Has parenthetical content
                base_label = parts[0].replace("_", " ")
                parenthetical = parts[1]
                label = f"{base_label} ({parenthetical}"
            else:
                # No parenthetical content
                label = label.replace("_", " ")
        
        simple_edges[(u, v)].append(label)
    
    # Join the labels for each edge
    labels = {k: ", ".join(vs) for k, vs in simple_edges.items()}
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# Batch/single pair processing helper
# ---------------------------------------------------------------------------
def check_existing_interaction_file(entity_a: str, entity_b: str, base: str) -> Optional[Path]:
    """Check if interaction data already exists for this gene pair."""
    # Check for exact match with current base pattern in current directory
    json_path = get_safe_path(base, "_interactions.json", use_cache=False)
    if json_path.exists():
        return json_path
    
    # Check in the INTERACTION_CACHE directory
    cache_path = get_safe_path(base, "_interactions.json", use_cache=True)
    if cache_path.exists():
        return cache_path
    
    # Base variations to check
    bases = [
        f"network_{entity_a}_{entity_b}",
        f"network_{entity_b}_{entity_a}",
        f"{entity_a}_{entity_b}",
        f"{entity_b}_{entity_a}"
    ]
    
    # Check both current directory and cache for all base variations
    for b in bases:
        # Check current directory
        local_path = get_safe_path(b, "_interactions.json", use_cache=False)
        if local_path.exists():
            return local_path
            
        # Check cache directory
        cache_path = get_safe_path(b, "_interactions.json", use_cache=True)
        if cache_path.exists():
            return cache_path
    
    # For more thorough searching, use regex pattern matching for exact gene names
    import re
    
    # Define regex patterns for exact matching of gene names in filenames
    # These patterns ensure gene names are properly bounded by underscores, start/end of string, or other delimiters
    patterns = [
        # network_A_B format
        re.compile(rf"network_{re.escape(entity_a)}_{re.escape(entity_b)}_interactions\.json$"),
        re.compile(rf"network_{re.escape(entity_b)}_{re.escape(entity_a)}_interactions\.json$"),
        # A_B format
        re.compile(rf"^{re.escape(entity_a)}_{re.escape(entity_b)}_interactions\.json$"),
        re.compile(rf"^{re.escape(entity_b)}_{re.escape(entity_a)}_interactions\.json$"),
        # Any path containing the exact gene pair pattern
        re.compile(rf"(^|[_-]){re.escape(entity_a)}[_-]{re.escape(entity_b)}([_-]|$).*interactions\.json$"),
        re.compile(rf"(^|[_-]){re.escape(entity_b)}[_-]{re.escape(entity_a)}([_-]|$).*interactions\.json$")
    ]
    
    # Search in current directory
    for file_path in Path(".").glob("*interactions.json"):
        file_name = file_path.name
        for pattern in patterns:
            if pattern.search(file_name):
                return file_path
    
    # Search in cache directory
    for file_path in INTERACTION_CACHE.glob("*interactions.json"):
        file_name = file_path.name
        for pattern in patterns:
            if pattern.search(file_name):
                return file_path
    
    return None

def print_interaction_summary(file_path: Path, entity_a: str, entity_b: str) -> None:
    """Load and print the interaction path summaries from an existing JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if "no_path" in data and data["no_path"]:
            print(f"[info] No plausible interaction path found for {entity_a} - {entity_b}: {data.get('reason', '(no reason provided)')}")
            return
            
        paths_list = data.get("paths", [])
        if not paths_list:
            print(f"[info] No paths found in the existing file for {entity_a} - {entity_b}.")
            return
            
        print(f"\n=== Interaction Path Summaries for {entity_a} -> {entity_b} (from cache) ===")
        for i, path in enumerate(paths_list, 1):
            desc = path.get("summary", "(no summary)")
            prob = path.get("overall_probability", "?")
            print(f"Path {i} (P={prob}): {desc}")
        print("")  # blank line
    except Exception as e:
        print(f"[warn] Failed to load or parse interaction data from {file_path}: {e}")

def process_pair(entity_a: str, entity_b: str, base: str, paths: int, model: str) -> None:
    # Check if this interaction has already been processed
    existing_file = check_existing_interaction_file(entity_a, entity_b, base)
    if existing_file:
        print(f"[info] SKIPPING: Interaction data for {entity_a} - {entity_b} already exists at {existing_file}")
        # Load and print the summary from the existing file
        print_interaction_summary(existing_file, entity_a, entity_b)
        return
    
    print(f"[info] Querying model {model} for interaction chains between {entity_a} and {entity_b}…")
    try:
        response = query_paths(entity_a, entity_b, paths, model)
    except Exception as exc:
        print(f"[error] Failed to query paths for {entity_a}, {entity_b}: {exc}", file=sys.stderr)
        return

    if response.get("no_path"):
        print(f"[warn] No plausible interaction path found for {entity_a} - {entity_b}: {response.get('reason', '(no reason provided)')}")
        return

    paths_list = response.get("paths", [])
    if not paths_list:
        print(f"[warn] Model returned zero paths for {entity_a} - {entity_b}.")
        return

    # Use the safe path function to avoid INTERACTION_CACHE duplication
    # Save raw JSON to cache directory
    json_path = get_safe_path(base, "_interactions.json", use_cache=True)
    json_path.parent.mkdir(exist_ok=True, parents=True)  # Ensure directory exists
    json_path.write_text(json.dumps(response, indent=2))
    print(f"[info] Wrote {json_path}")

    # Also save to current directory for backward compatibility
    local_json_path = get_safe_path(base, "_interactions.json", use_cache=False)
    local_json_path.write_text(json.dumps(response, indent=2))
    print(f"[info] Wrote {local_json_path}")

    # Build graph
    G = build_graph(paths_list)
    
    # Save DOT file to cache
    dot_path = get_safe_path(base, ".dot", use_cache=True)
    try:
        dump_graphviz(G, dot_path)
        print(f"[info] Wrote Graphviz file {dot_path}")
    except Exception as exc:
        print(f"[warn] Could not write DOT file {dot_path}: {exc}")
        
    # Save DOT file to current directory too
    local_dot_path = get_safe_path(base, ".dot", use_cache=False)
    try:
        dump_graphviz(G, local_dot_path)
        print(f"[info] Wrote Graphviz file {local_dot_path}")
    except Exception as exc:
        print(f"[warn] Could not write DOT file {local_dot_path}: {exc}")

    # Save PNG to cache
    png_path = get_safe_path(base, ".png", use_cache=True)
    try:
        draw_graph_png(G, png_path)
        print(f"[info] Wrote PNG {png_path}")
    except Exception as exc:
        print(f"[warn] Could not render PNG {png_path}: {exc}")
        
    # Save PNG to current directory too
    local_png_path = get_safe_path(base, ".png", use_cache=False)
    try:
        draw_graph_png(G, local_png_path)
        print(f"[info] Wrote PNG {local_png_path}")
    except Exception as exc:
        print(f"[warn] Could not render PNG {local_png_path}: {exc}")

    # Summaries
    print(f"\n=== Interaction Path Summaries for {entity_a} -> {entity_b} ===")
    for i, path in enumerate(paths_list, 1):
        desc = path.get("summary", "(no summary)")
        prob = path.get("overall_probability", "?")
        print(f"Path {i} (P={prob}): {desc}")
    print("")  # blank line

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Discover interaction chains between two genes/proteins using GPT‑4. Skips pairs with existing analysis.")
    parser.add_argument("entity_a", nargs="?", help="First gene or protein identifier (e.g. TP53)")
    parser.add_argument("entity_b", nargs="?", help="Second gene or protein identifier (e.g. EGFR)")
    parser.add_argument("--input-file", "-i", dest="input_file", help="Path to input file with gene/protein pairs, one per line")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name (default: gpt-4.1)")
    parser.add_argument("--paths", type=int, default=3, help="Maximum number of paths to request (default: 3)")
    parser.add_argument("--out", default="network", help="Output file prefix (default: network)")
    args = parser.parse_args()

    # Ensure API key
    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Process either batch from input file or single pair
    if args.input_file:
        try:
            with open(args.input_file) as f:
                lines = f.readlines()
        except Exception as exc:
            print(f"[error] Could not open input file {args.input_file}: {exc}", file=sys.stderr)
            sys.exit(1)
        
        # Track statistics
        total_pairs = 0
        skipped_pairs = 0
        processed_pairs = 0
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = re.split(r'[,\s]+', line)
            if len(parts) < 2:
                print(f"[warn] Skipping invalid line: {line}")
                continue
            
            total_pairs += 1
            ent_a, ent_b = parts[0], parts[1]
            base = f"{args.out}_{ent_a}_{ent_b}"
            
            # Check if this interaction already exists before processing
            existing_file = check_existing_interaction_file(ent_a, ent_b, base)
            if existing_file:
                print(f"[info] SKIPPING: Interaction data for {ent_a} - {ent_b} already exists at {existing_file}")
                # Load and print the summary from the existing file
                print_interaction_summary(existing_file, ent_a, ent_b)
                skipped_pairs += 1
                continue
                
            process_pair(ent_a, ent_b, base, args.paths, args.model)
            processed_pairs += 1
        
        # Print summary statistics
        print(f"\n[summary] Total pairs: {total_pairs}")
        print(f"[summary] Skipped existing pairs: {skipped_pairs}")
        print(f"[summary] Newly processed pairs: {processed_pairs}")
        sys.exit(0)
    elif args.entity_a and args.entity_b:
        # For single pair mode, also check for existing file
        base = args.out
        existing_file = check_existing_interaction_file(args.entity_a, args.entity_b, base)
        if existing_file:
            print(f"[info] SKIPPING: Interaction data for {args.entity_a} - {args.entity_b} already exists at {existing_file}")
            print(f"[info] To force regeneration, delete {existing_file} and run again.")
            # Load and print the summary from the existing file
            print_interaction_summary(existing_file, args.entity_a, args.entity_b)
        else:
            process_pair(args.entity_a, args.entity_b, base, args.paths, args.model)
    else:
        parser.error("Must provide either entity_a and entity_b or --input-file")


if __name__ == "__main__":
    main()
