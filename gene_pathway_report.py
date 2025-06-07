#!/usr/bin/env python3
"""
Gene Pathway Report Generator
============================

This script generates comprehensive reports on gene interaction pathways by:
1. Checking the INTERACTION_CACHE for existing gene pair interaction data
2. If not found, calling gene_chain_v1.py to generate the data
3. Creating detailed reports that distinguish between well-supported interactions 
   and conjectural ones

Usage:
------
$ export OPENAI_API_KEY="sk-..."
$ python gene_pathway_report.py TP53 EGFR --model gpt-4.1 --paths 4 --output report.md
$ python gene_pathway_report.py --input-file pairs.txt --model gpt-4.1 --output report.md

Requirements:
------------
- Python ≥3.8
- openai
- All dependencies required by gene_chain_v1.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import openai

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

# ---------------------------------------------------------------------------
# File handling utilities
# ---------------------------------------------------------------------------

def find_interaction_file(entity_a: str, entity_b: str, base: str = "network") -> Optional[Path]:
    """
    Find an existing interaction file for the given gene pair.
    Checks both the current directory and the INTERACTION_CACHE.
    """
    # Check all possible variations of the filename
    patterns = [
        f"{base}_{entity_a}_{entity_b}_interactions.json",
        f"{base}_{entity_b}_{entity_a}_interactions.json",
        f"network_{entity_a}_{entity_b}_interactions.json",
        f"network_{entity_b}_{entity_a}_interactions.json",
        f"{entity_a}_{entity_b}_interactions.json",
        f"{entity_b}_{entity_a}_interactions.json"
    ]
    
    # First check the cache directory
    for pattern in patterns:
        cache_path = INTERACTION_CACHE / pattern
        if cache_path.exists():
            return cache_path
    
    # Then check the current directory
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            return path
    
    return None

def generate_interaction_data(entity_a: str, entity_b: str, model: str, paths: int, config_file: Optional[str] = None) -> Path:
    """
    Generate interaction data for a gene pair by calling gene_chain_v1.py.
    Returns the path to the generated JSON file.
    """
    print(f"[info] Generating interaction data for {entity_a}-{entity_b} using gene_chain_v1.py...")
    
    # Prepare the base name for output files (without directory or suffix)
    base = f"network_{entity_a}_{entity_b}"
    
    # Call gene_chain_v1.py to generate the data
    # Use base name only without the cache directory - gene_chain_v1.py already uses INTERACTION_CACHE
    
    # If model_config is available, check for model configuration
    model_alias = model
    if MODEL_CONFIG_AVAILABLE:
        try:
            # Check if model is in configuration by name or shortname
            model_cfg = None
            try:
                model_cfg = get_model_config(model, config_file=config_file)
            except ValueError:
                try:
                    model_cfg = get_model_config(model, by_shortname=True, config_file=config_file)
                except ValueError:
                    pass
                    
            if model_cfg:
                # Use the configured model name
                print(f"[info] Using model configuration: {model} -> {model_cfg['openai_model']}")
                model_alias = model_cfg["openai_model"]
        except Exception as e:
            print(f"[warn] Error looking up model configuration: {e}")
    
    cmd = [
        sys.executable, 
        "gene_chain_v1.py", 
        entity_a, 
        entity_b,
        "--model", model_alias,
        "--paths", str(paths),
        "--out", base  # Just use the base name, not a path inside INTERACTION_CACHE
    ]
    
    try:
        # Set check=False to prevent CalledProcessError from being raised
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        # Check if gene_chain_v1.py reported no path found
        if "No plausible interaction path found" in result.stdout or "No plausible interaction path found" in result.stderr:
            print(f"[info] No interaction paths found between {entity_a} and {entity_b}")
            # Return a special value to indicate no path found
            return "NO_PATH_FOUND"
            
        # Check if there was a failure not related to "no path found"
        if result.returncode != 0:
            error_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
            
            # Check if this is a "no paths" error vs other error
            if "No plausible interaction path found" in error_output or "Model returned zero paths" in error_output:
                print(f"[info] No interaction paths found between {entity_a} and {entity_b}")
                return "NO_PATH_FOUND"
            else:
                print(f"[error] gene_chain_v1.py returned non-zero exit code: {result.returncode}", file=sys.stderr)
                print(error_output, file=sys.stderr)
                # Continue with processing other pairs rather than exiting
                raise ValueError(f"Command failed with exit code {result.returncode}")
        
        # After execution, use find_interaction_file to locate the generated file
        interaction_file = find_interaction_file(entity_a, entity_b, base)
        
        if interaction_file:
            print(f"[info] Successfully generated interaction data: {interaction_file}")
            return interaction_file
        else:
            # Sleep a moment to ensure file system sync
            time.sleep(1)
            
            # Try again with more aggressive pattern matching
            interaction_file = find_interaction_file(entity_a, entity_b)
            
            if interaction_file:
                print(f"[info] Found interaction data: {interaction_file}")
                return interaction_file
            
            # Final fallback - direct glob search
            print(f"[warn] Data generation completed but output file not found in expected locations", file=sys.stderr)
            potential_files = []
            # Check standard locations
            for location in [Path("."), INTERACTION_CACHE]:
                potential_files.extend(list(location.glob(f"*{entity_a}*{entity_b}*interactions*.json")))
                potential_files.extend(list(location.glob(f"*{entity_b}*{entity_a}*interactions*.json")))
                
            if potential_files:
                found_file = potential_files[0]
                print(f"[info] Found alternative output file: {found_file}")
                return found_file
            
            # If process completed but no file found, assume no paths were found
            return "NO_PATH_FOUND"
    except Exception as e:
        print(f"[error] Error processing {entity_a}-{entity_b}: {e}", file=sys.stderr)
        # Return a special error marker rather than exiting
        return "ERROR_PROCESSING"

def load_interaction_data(file_path: Path) -> Dict[str, Any]:
    """Load interaction data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[error] Failed to load interaction data from {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def classify_evidence(evidence: str) -> Tuple[str, bool]:
    """
    Classify evidence as well-supported or conjectural.
    Returns a tuple of (formatted evidence, is_well_supported).
    """
    # Check for well-supported evidence (PMIDs, database references)
    has_pmid = re.search(r'PMID:?\s*\d+', evidence, re.IGNORECASE) is not None
    has_database = any(db in evidence.upper() for db in ["UNIPROT", "KEGG", "REACTOME", "GO:", "GENE ONTOLOGY", "STRINGDB", "STRING-DB", "BIOGRID"])
    has_doi = "DOI:" in evidence or "doi.org" in evidence
    
    is_well_supported = has_pmid or has_database or has_doi
    
    # Format the evidence string
    if is_well_supported:
        if has_pmid:
            matches = re.findall(r'(?:PMID:?\s*(\d+))', evidence, re.IGNORECASE)
            for pmid in matches:
                evidence = evidence.replace(f"PMID: {pmid}", f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                evidence = evidence.replace(f"PMID:{pmid}", f"[PMID: {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
    
    return evidence, is_well_supported

def analyze_interaction_paths(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze interaction paths, classifying each interaction as well-supported or conjectural.
    Returns a structured analysis of the paths.
    """
    paths = data.get("paths", [])
    if not paths:
        return {"error": "No interaction paths found in the data"}
    
    analyzed_paths = []
    
    for i, path in enumerate(paths, 1):
        edges = path.get("edges", [])
        summary = path.get("summary", "No summary provided")
        overall_prob = path.get("overall_probability", 0)
        
        analyzed_edges = []
        supported_edges = 0
        conjectural_edges = 0
        
        for edge in edges:
            source = edge.get("source", "Unknown")
            target = edge.get("target", "Unknown")
            mechanism = edge.get("mechanism", "Unknown")
            probability = float(edge.get("probability", 0))
            raw_evidence = edge.get("evidence", "No evidence provided")
            
            evidence, is_well_supported = classify_evidence(raw_evidence)
            
            if is_well_supported:
                supported_edges += 1
            else:
                conjectural_edges += 1
            
            analyzed_edges.append({
                "source": source,
                "target": target,
                "mechanism": mechanism,
                "probability": probability,
                "evidence": evidence,
                "is_well_supported": is_well_supported
            })
        
        # Calculate the path support ratio
        total_edges = len(edges)
        support_ratio = supported_edges / total_edges if total_edges > 0 else 0
        
        # Classify the path as a whole
        if support_ratio >= 0.7:
            path_classification = "Well-supported"
        elif support_ratio >= 0.3:
            path_classification = "Partially supported"
        else:
            path_classification = "Primarily conjectural"
        
        analyzed_paths.append({
            "path_number": i,
            "summary": summary,
            "overall_probability": overall_prob,
            "support_ratio": support_ratio,
            "supported_edges": supported_edges,
            "conjectural_edges": conjectural_edges,
            "classification": path_classification,
            "edges": analyzed_edges
        })
    
    return {"paths": analyzed_paths}

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(entity_a: str, entity_b: str, analysis: Dict[str, Any]) -> str:
    """Generate a Markdown report for the interaction pathways between two genes/proteins."""
    paths = analysis.get("paths", [])
    if not paths:
        return f"# No Interaction Pathways Found Between {entity_a} and {entity_b}\n\nNo valid interaction pathways were found in the data."
    
    # Start building the report
    report = [
        f"# Interaction Pathways: {entity_a} → {entity_b}",
        f"\nThis report contains analyzed interaction pathways between {entity_a} and {entity_b}.",
        "Each pathway is classified based on its supporting evidence.\n",
        "## Summary of Pathways\n"
    ]
    
    # Add a summary table
    report.append("| Path | Classification | Support Ratio | Overall Probability | Summary |")
    report.append("|------|---------------|--------------|---------------------|---------|")
    
    for path in paths:
        path_num = path["path_number"]
        classification = path["classification"]
        support_ratio = f"{path['support_ratio']:.2f} ({path['supported_edges']}/{path['supported_edges'] + path['conjectural_edges']})"
        overall_prob = f"{path['overall_probability']:.2f}"
        summary = path["summary"]
        
        report.append(f"| {path_num} | {classification} | {support_ratio} | {overall_prob} | {summary} |")
    
    report.append("\n## Detailed Pathway Analysis\n")
    
    # Add detailed path information
    for path in paths:
        path_num = path["path_number"]
        classification = path["classification"]
        
        report.append(f"### Path {path_num}: {classification}")
        report.append(f"\n**Summary**: {path['summary']}")
        report.append(f"\n**Overall Probability**: {path['overall_probability']:.2f}")
        report.append(f"\n**Support Ratio**: {path['support_ratio']:.2f} ({path['supported_edges']}/{path['supported_edges'] + path['conjectural_edges']} edges supported)\n")
        
        # Add edge details
        report.append("#### Interactions in this pathway\n")
        
        for i, edge in enumerate(path["edges"], 1):
            source = edge["source"]
            target = edge["target"]
            mechanism = edge["mechanism"]
            probability = edge["probability"]
            evidence = edge["evidence"]
            is_supported = edge["is_well_supported"]
            
            evidence_class = "Well-supported" if is_supported else "Conjectural"
            evidence_marker = "✓" if is_supported else "?"
            
            report.append(f"**Interaction {i}**: {source} → {target} ({evidence_marker})")
            report.append(f"\n- **Mechanism**: {mechanism}")
            report.append(f"- **Probability**: {probability:.2f}")
            report.append(f"- **Evidence** ({evidence_class}): {evidence}\n")
        
        report.append("---\n")
    
    report.append("\n## Methodology\n")
    report.append("This report was generated using Gene Chain, an AI-driven system that uses GPT-4 to discover and analyze possible interaction pathways between genes and proteins.")
    report.append("\nPathways are classified as:")
    report.append("- **Well-supported**: ≥70% of interactions have references to scientific literature (PMID) or established databases")
    report.append("- **Partially supported**: 30-70% of interactions have strong supporting evidence")
    report.append("- **Primarily conjectural**: <30% of interactions have strong supporting evidence")
    report.append("\nInteractions marked with ✓ have explicit references to literature or databases, while those marked with ? are more speculative.")
    report.append("\nEven well-supported pathways should be verified with targeted experiments or further literature review.")
    
    return "\n".join(report)

def process_and_report_pair(entity_a: str, entity_b: str, model: str, paths: int, config_file: Optional[str] = None) -> str:
    """Process a gene pair and generate a report."""
    # Check if interaction data already exists
    interaction_file = find_interaction_file(entity_a, entity_b)
    
    if interaction_file:
        print(f"[info] Found existing interaction data: {interaction_file}")
    else:
        # Generate new interaction data
        interaction_file = generate_interaction_data(entity_a, entity_b, model, paths, config_file)
    
    # Handle special return values
    if interaction_file == "NO_PATH_FOUND":
        # Generate a "no path found" report
        return f"# No Interaction Pathways Found: {entity_a} → {entity_b}\n\n" + \
               f"After analysis, no plausible interaction pathways were found between {entity_a} and {entity_b}.\n\n" + \
               "This could be due to:\n" + \
               "- Lack of documented interactions in the literature\n" + \
               "- Interactions that are too distant or indirect to reliably trace\n" + \
               "- Genes/proteins that function in separate biological pathways\n\n" + \
               "Consider searching for intermediate connecting genes or exploring broader pathway databases."
    
    if interaction_file == "ERROR_PROCESSING":
        # Generate an error report
        return f"# Error Processing: {entity_a} → {entity_b}\n\n" + \
               f"An error occurred while attempting to analyze interactions between {entity_a} and {entity_b}.\n\n" + \
               "See the console output for more details about the error."
    
    # Load and analyze the interaction data
    try:
        data = load_interaction_data(interaction_file)
        analysis = analyze_interaction_paths(data)
        
        # Generate the report
        return generate_markdown_report(entity_a, entity_b, analysis)
    except Exception as e:
        print(f"[error] Failed to analyze or generate report for {entity_a}-{entity_b}: {e}", file=sys.stderr)
        return f"# Error Analyzing: {entity_a} → {entity_b}\n\n" + \
               f"Successfully generated interaction data, but encountered an error during analysis: {str(e)}"

def process_batch(input_file: str, model: str, paths: int, config_file: Optional[str] = None) -> Dict[str, str]:
    """Process multiple gene pairs from an input file."""
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[error] Failed to read input file {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Filter out empty lines and comments
    valid_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    
    reports = {}
    pairs_processed = 0
    errors = 0
    skipped = 0
    
    for i, line in enumerate(valid_lines):
        parts = re.split(r'[,\s]+', line)
        if len(parts) < 2:
            print(f"[warn] Skipping invalid line: {line}")
            skipped += 1
            continue
        
        entity_a, entity_b = parts[0], parts[1]
        pair_key = f"{entity_a}_{entity_b}"
        
        print(f"\n[info] Processing gene pair: {entity_a}-{entity_b} ({i+1}/{len(valid_lines)})")
        
        try:
            # First check if we already have this report
            if pair_key in reports:
                print(f"[info] Report for {entity_a}-{entity_b} already generated, skipping")
                skipped += 1
                continue
                
            report = process_and_report_pair(entity_a, entity_b, model, paths, config_file)
            reports[pair_key] = report
            pairs_processed += 1
            
            # Every 10 pairs, print status update
            if pairs_processed % 10 == 0:
                print(f"\n[status] Progress: {pairs_processed} pairs processed, {errors} errors, {skipped} skipped, {len(valid_lines) - i - 1} remaining")
                
        except Exception as e:
            print(f"[error] Failed to process {entity_a}-{entity_b}: {e}", file=sys.stderr)
            errors += 1
            # Add error placeholder to reports if specified
            reports[pair_key] = f"# Error Processing {entity_a}-{entity_b}\n\nAn error occurred while processing this gene pair: {str(e)}"
    
    print(f"\n[summary] Total gene pairs: {len(valid_lines)}")
    print(f"[summary] Successfully processed: {pairs_processed}")
    print(f"[summary] Errors: {errors}")
    print(f"[summary] Skipped: {skipped}")
    
    return reports

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive reports on gene interaction pathways"
    )
    parser.add_argument("entity_a", nargs="?", help="First gene or protein identifier (e.g. TP53)")
    parser.add_argument("entity_b", nargs="?", help="Second gene or protein identifier (e.g. EGFR)")
    parser.add_argument("--input-file", "-i", dest="input_file", 
                        help="Path to input file with gene/protein pairs, one per line")
    parser.add_argument("--model", default="gpt-4.1", 
                        help="OpenAI model name or shortname from model_servers.yaml (default: gpt-4.1)")
    parser.add_argument("--model-by-shortname", action="store_true",
                        help="Interpret model argument as a shortname instead of full model name")
    parser.add_argument("--config-file", 
                        help="Path to model configuration YAML file (default: model_servers.yaml)")
    parser.add_argument("--config", dest="config_file_alt",
                        help="Path to model configuration file (alias for --config-file)")                 
    parser.add_argument("--paths", type=int, default=3, 
                        help="Maximum number of paths to request (default: 3)")
    parser.add_argument("--output", "-o", dest="output_file", 
                        help="Output file (default: gene_pathway_report.md for single pair or multiple_pathway_report.md for multiple pairs)")
    args = parser.parse_args()
    
    # Ensure API key if using OpenAI directly (might not be needed if using model_config)
    if not os.getenv("OPENAI_API_KEY") and not MODEL_CONFIG_AVAILABLE:
        print("[error] OPENAI_API_KEY environment variable is not set and model_config is not available.", file=sys.stderr)
        sys.exit(1)
        
    # Determine which config file to use (--config takes precedence over --config-file)
    config_file_to_use = args.config_file_alt or args.config_file
    
    # If model configuration is available, try to load the config file
    if MODEL_CONFIG_AVAILABLE and config_file_to_use:
        try:
            # Import config_file parameter
            from model_config import load_config
            load_config(config_file_to_use)
            print(f"[info] Loaded model configuration from {config_file_to_use}")
        except Exception as e:
            print(f"[warn] Failed to load model configuration from {config_file_to_use}: {e}", file=sys.stderr)
            
    # If using shortname, adjust the model parameter
    model = args.model
    if MODEL_CONFIG_AVAILABLE and args.model_by_shortname:
        try:
            model_cfg = get_model_config(args.model, by_shortname=True, config_file=config_file_to_use)
            print(f"[info] Using model '{model_cfg['openai_model']}' from shortname '{args.model}'")
            model = model_cfg["openai_model"]
        except Exception as e:
            print(f"[warn] Failed to resolve model shortname '{args.model}': {e}", file=sys.stderr)
    
    # Process either a single pair or batch from input file
    if args.input_file:
        # Process multiple pairs from input file
        output_file = args.output_file or "multiple_pathway_report.md"
        reports = process_batch(args.input_file, model, args.paths, config_file_to_use)
        
        # Combine all reports into a single document
        combined_report = ["# Multiple Gene Pathway Interaction Report\n"]
        for pair_key, report in reports.items():
            # Skip the methodology section for all but the last report
            if pair_key != list(reports.keys())[-1]:
                report_without_methodology = "\n".join(report.split("\n## Methodology")[0].split("\n"))
                combined_report.append(report_without_methodology)
                combined_report.append("\n\n---\n\n")
            else:
                combined_report.append(report)
        
        # Write the combined report
        with open(output_file, 'w') as f:
            f.write("\n".join(combined_report))
        print(f"\n[info] Wrote combined report to {output_file}")
        
    elif args.entity_a and args.entity_b:
        # Process a single pair
        output_file = args.output_file or f"gene_pathway_report_{args.entity_a}_{args.entity_b}.md"
        report = process_and_report_pair(args.entity_a, args.entity_b, model, args.paths, config_file_to_use)
        
        # Write the report
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n[info] Wrote report to {output_file}")
        
    else:
        parser.error("Must provide either entity_a and entity_b or --input-file")

if __name__ == "__main__":
    main()