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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

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

def gpt_call(messages: List[Dict[str, str]], *, model: str = "gpt-4o", max_tokens: int = 2048,
             temperature: float = 0.2, retries: int = 3, backoff: float = 5.0) -> str:
    """Robust wrapper around openai.ChatCompletion.create with simple retry."""
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
        return json.loads(raw)
    except json.JSONDecodeError as exc:
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
            mech = edge["mechanism"]
            prob = float(edge["probability"])
            evid = edge.get("evidence", "?")
            G.add_edge(src, tgt, label=mech, probability=prob, evidence=evid)
    return G


def dump_graphviz(G: nx.MultiDiGraph, filepath: Path):
    """Write the graph to *filepath* in Graphviz dot format."""
    nx.drawing.nx_pydot.write_dot(G, str(filepath))


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
        simple_edges[(u, v)].append(d["label"])
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
def process_pair(entity_a: str, entity_b: str, base: str, paths: int, model: str) -> None:
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

    # Save raw JSON
    json_path = Path(f"{base}_interactions.json")
    json_path.write_text(json.dumps(response, indent=2))
    print(f"[info] Wrote {json_path}")

    # Build graph
    G = build_graph(paths_list)
    dot_path = Path(f"{base}.dot")
    try:
        dump_graphviz(G, dot_path)
        print(f"[info] Wrote Graphviz file {dot_path}")
    except Exception as exc:
        print(f"[warn] Could not write DOT file {dot_path}: {exc}")

    png_path = Path(f"{base}.png")
    try:
        draw_graph_png(G, png_path)
        print(f"[info] Wrote PNG {png_path}")
    except Exception as exc:
        print(f"[warn] Could not render PNG {png_path}: {exc}")

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
    parser = argparse.ArgumentParser(description="Discover interaction chains between two genes/proteins using GPT‑4.")
    parser.add_argument("entity_a", nargs="?", help="First gene or protein identifier (e.g. TP53)")
    parser.add_argument("entity_b", nargs="?", help="Second gene or protein identifier (e.g. EGFR)")
    parser.add_argument("--input-file", "-i", dest="input_file", help="Path to input file with gene/protein pairs, one per line")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
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
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r'[,\s]+', line)
            if len(parts) < 2:
                print(f"[warn] Skipping invalid line: {line}")
                continue
            ent_a, ent_b = parts[0], parts[1]
            base = f"{args.out}_{ent_a}_{ent_b}"
            process_pair(ent_a, ent_b, base, args.paths, args.model)
        sys.exit(0)
    elif args.entity_a and args.entity_b:
        process_pair(args.entity_a, args.entity_b, args.out, args.paths, args.model)
    else:
        parser.error("Must provide either entity_a and entity_b or --input-file")


if __name__ == "__main__":
    main()
