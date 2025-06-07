#!/usr/bin/env python3
"""
Compare Pathway Analyses Across Multiple Files

This script analyzes multiple *-TF_pathways.json files to identify:
1. Common pathways across all files
2. Pathways common to specific subsets of files
3. Unique pathways found in only one file

It produces summary statistics and detailed output files for each set of pathways.

Usage:
------
$ python compare_pathways.py [--input-pattern PATTERN] [--output-dir OUTPUT_DIR]

Arguments:
  --input-pattern PATTERN    Glob pattern for input files (default: *-TF_pathways.json)
  --output-dir OUTPUT_DIR    Directory for output files (default: pathway_comparisons)
  --summary-only             Only output summary statistics (no detailed files)
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Any, Set, Tuple

def load_files(pattern: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all files matching the pattern."""
    file_paths = glob.glob(pattern)
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(file_paths)} files: {', '.join(file_paths)}")
    
    files_data = {}
    for file_path in file_paths:
        # Extract file identifier (e.g., "A" from "A-TF_pathways.json")
        file_id = os.path.basename(file_path).split('-')[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            files_data[file_id] = data
            print(f"Loaded {len(data)} gene pairs from {file_path}")
    
    return files_data

def create_pathway_signature(pathway: Dict[str, Any]) -> str:
    """
    Create a unique signature for a pathway to enable comparison.
    The signature combines key elements of the pathway that identify its core structure.
    """
    # Extract key components that identify the pathway
    edges = []
    for interaction in pathway.get("interactions", []):
        edge = f"{interaction['source']}->{interaction['target']}:{interaction['mechanism']}"
        edges.append(edge)
    
    # Sort edges for consistent comparison
    edges.sort()
    
    # Create signature as a combination of sorted edges
    signature = "||".join(edges)
    return signature

def create_gene_pair_key(gene_pair: Dict[str, Any]) -> str:
    """Create a key for a gene pair to identify it."""
    return f"{gene_pair['source_gene']}->{gene_pair['target_gene']}"

def analyze_pathways(files_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze pathways across all files to identify common and unique pathways.
    Returns a dictionary with analysis results.
    """
    # Dictionary to store pathway signatures mapped to their occurrences
    pathway_signatures = defaultdict(lambda: {"files": set(), "pathways": []})
    
    # Dictionary to store gene pair occurrences across files
    gene_pair_occurrences = defaultdict(set)
    
    # Process each file
    for file_id, gene_pairs in files_data.items():
        # Track gene pairs in this file
        file_gene_pairs = set()
        
        for gene_pair in gene_pairs:
            gene_pair_key = create_gene_pair_key(gene_pair)
            file_gene_pairs.add(gene_pair_key)
            
            for pathway in gene_pair.get("pathways", []):
                signature = create_pathway_signature(pathway)
                
                # Add this pathway occurrence
                pathway_signatures[signature]["files"].add(file_id)
                
                # Store the full pathway data only once
                if not pathway_signatures[signature]["pathways"]:
                    pathway_copy = dict(pathway)
                    pathway_copy["source_gene"] = gene_pair["source_gene"]
                    pathway_copy["target_gene"] = gene_pair["target_gene"]
                    pathway_signatures[signature]["pathways"].append(pathway_copy)
        
        # Update gene pair occurrences
        for gene_pair_key in file_gene_pairs:
            gene_pair_occurrences[gene_pair_key].add(file_id)
    
    # Categorize pathways by their occurrence pattern
    occurrence_patterns = defaultdict(list)
    
    for signature, info in pathway_signatures.items():
        files_key = tuple(sorted(info["files"]))
        occurrence_patterns[files_key].extend(info["pathways"])
    
    # Categorize gene pairs by their occurrence pattern
    gene_pair_patterns = defaultdict(set)
    
    for gene_pair, files in gene_pair_occurrences.items():
        files_key = tuple(sorted(files))
        gene_pair_patterns[files_key].add(gene_pair)
    
    # Prepare the final analysis result
    result = {
        "pathway_patterns": dict(occurrence_patterns),
        "gene_pair_patterns": {k: list(v) for k, v in gene_pair_patterns.items()},
        "file_ids": sorted(files_data.keys()),
        "total_unique_pathways": len(pathway_signatures),
        "total_unique_gene_pairs": len(gene_pair_occurrences)
    }
    
    return result

def create_pattern_description(pattern: Tuple[str, ...], all_files: List[str]) -> str:
    """Create a human-readable description of a file pattern."""
    if len(pattern) == len(all_files):
        return "Common to all files"
    elif len(pattern) == 1:
        return f"Unique to {pattern[0]}"
    else:
        return f"Common to {', '.join(pattern)}"

def generate_gene_pair_summary(analysis: Dict[str, Any], files_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of each gene pair and their pathways across all files.
    Returns a dictionary mapping each gene pair to its pathway information.
    """
    gene_pair_summary = {}
    all_files = analysis["file_ids"]
    
    # First, collect all pathways for each gene pair from each file
    for file_id, gene_pairs in files_data.items():
        for gene_pair in gene_pairs:
            key = f"{gene_pair['source_gene']}->{gene_pair['target_gene']}"
            
            if key not in gene_pair_summary:
                gene_pair_summary[key] = {
                    "source": gene_pair['source_gene'],
                    "target": gene_pair['target_gene'],
                    "files": set(),
                    "pathways_by_file": {},
                    "total_pathways": 0
                }
            
            gene_pair_summary[key]["files"].add(file_id)
            gene_pair_summary[key]["pathways_by_file"][file_id] = gene_pair.get("pathways", [])
            gene_pair_summary[key]["total_pathways"] += len(gene_pair.get("pathways", []))
    
    # Convert sets to lists for better display
    for key in gene_pair_summary:
        gene_pair_summary[key]["files"] = sorted(gene_pair_summary[key]["files"])
    
    return gene_pair_summary

def generate_summary(analysis: Dict[str, Any], files_data: Dict[str, List[Dict[str, Any]]], 
                     output_dir: str, summary_only: bool = False) -> None:
    """Generate summary statistics and detailed output files."""
    all_files = analysis["file_ids"]
    
    # Create output directory if it doesn't exist
    if not summary_only and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Summary statistics
    print("\n===== PATHWAY ANALYSIS SUMMARY =====\n")
    print(f"Total Files Analyzed: {len(all_files)}")
    print(f"Total Unique Pathways: {analysis['total_unique_pathways']}")
    print(f"Total Unique Gene Pairs: {analysis['total_unique_gene_pairs']}")
    print("\nPathway Distribution:")
    
    # Sort patterns by number of files (descending) then by file identifiers
    sorted_patterns = sorted(
        analysis["pathway_patterns"].keys(),
        key=lambda p: (-len(p), p)
    )
    
    # Prepare detailed information for each pattern
    for pattern in sorted_patterns:
        pathways = analysis["pathway_patterns"][pattern]
        description = create_pattern_description(pattern, all_files)
        
        # Print summary statistics for this pattern
        print(f"  - {description}: {len(pathways)} pathways")
        
        # Skip detailed file creation if summary only
        if summary_only:
            continue
            
        # Create detailed file for this pattern
        pattern_desc = "_".join(pattern) if pattern else "no_common"
        output_file = os.path.join(output_dir, f"pathways_{pattern_desc}.json")
        
        # Group pathways by gene pair
        grouped_pathways = defaultdict(list)
        for pathway in pathways:
            gene_pair = f"{pathway['source_gene']}->{pathway['target_gene']}"
            # Remove the source_gene and target_gene keys from the pathway
            pathway_copy = {k: v for k, v in pathway.items() 
                            if k not in ["source_gene", "target_gene"]}
            grouped_pathways[gene_pair].append(pathway_copy)
        
        # Format for output
        output_data = {
            "pattern": pattern,
            "description": description,
            "count": len(pathways),
            "gene_pairs": {
                gene_pair: {"pathways": paths}
                for gene_pair, paths in grouped_pathways.items()
            }
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"      Saved to {output_file}")
    
    # Generate and output gene pair summary
    gene_pair_summary = generate_gene_pair_summary(analysis, files_data)
    
    print("\n===== GENE PAIR SUMMARY =====\n")
    
    # Sort gene pairs by total number of pathways (descending)
    sorted_gene_pairs = sorted(
        gene_pair_summary.keys(),
        key=lambda pair: (-gene_pair_summary[pair]["total_pathways"], pair)
    )
    
    for gene_pair in sorted_gene_pairs:
        info = gene_pair_summary[gene_pair]
        source = info["source"]
        target = info["target"]
        files_present = ", ".join(info["files"])
        total_pathways = info["total_pathways"]
        
        print(f"\n{source} → {target}:")
        print(f"  Present in {len(info['files'])}/{len(all_files)} files: {files_present}")
        print(f"  Total pathways across all files: {total_pathways}")
        
        # Print pathway summary for each file
        for file_id in sorted(info["pathways_by_file"].keys()):
            pathways = info["pathways_by_file"][file_id]
            if not pathways:
                continue
                
            print(f"  File {file_id} ({len(pathways)} pathways):")
            for i, pathway in enumerate(pathways, 1):
                summary = pathway.get("summary", "No summary available")
                probability = pathway.get("overall_probability", "N/A")
                classification = pathway.get("classification", "Unknown")
                print(f"    {i}. {summary} (P={probability}, {classification})")
    
    # Also create a master summary file
    if not summary_only:
        summary_file = os.path.join(output_dir, "summary.json")
        summary_data = {
            "files_analyzed": all_files,
            "total_unique_pathways": analysis["total_unique_pathways"],
            "total_unique_gene_pairs": analysis["total_unique_gene_pairs"],
            "pattern_counts": {
                "_".join(pattern) if pattern else "no_common": len(pathways)
                for pattern, pathways in analysis["pathway_patterns"].items()
            },
            "gene_pair_pattern_counts": {
                "_".join(pattern) if pattern else "no_common": len(gene_pairs)
                for pattern, gene_pairs in analysis["gene_pair_patterns"].items()
            },
            "gene_pair_summary": gene_pair_summary
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\nMaster summary saved to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare pathway analyses across multiple files")
    parser.add_argument("--input-pattern", default="*-TF_pathways.json",
                        help="Glob pattern for input files (default: *-TF_pathways.json)")
    parser.add_argument("--output-dir", default="pathway_comparisons",
                        help="Directory for output files (default: pathway_comparisons)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only output summary statistics (no detailed files)")
    args = parser.parse_args()
    
    try:
        # Load all files
        files_data = load_files(args.input_pattern)
        
        # Analyze pathways
        analysis = analyze_pathways(files_data)
        
        # Generate summary and detailed files
        generate_summary(analysis, files_data, args.output_dir, args.summary_only)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()