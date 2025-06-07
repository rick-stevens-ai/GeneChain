#!/usr/bin/env python3
"""
Extract Pathway Information from a Markdown File

This script extracts pathway information from a specified Markdown file.
It extracts the gene pair, pathway summaries, and detailed pathway interactions,
outputting them in a structured format.

Usage:
------
$ python extract_pathways.py INPUT_FILE [--output OUTPUT_FILE] [--format {json,csv,tsv,md}]

Arguments:
  INPUT_FILE              The input markdown file to extract pathways from
  --output OUTPUT_FILE    Output file path (default: uses input filename with new extension)
  --format FORMAT         Output format: json, csv, tsv, or md (default: json)
  --include-evidence      Include evidence details in the output (default: False)
  --path-limit N          Limit the number of pathways per gene pair (default: no limit)
"""

import argparse
import glob
import json
import os
import re
import csv
from typing import Dict, List, Any, Optional, Tuple


def extract_pathways(file_path: str, include_evidence: bool = False, path_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Extract pathway information from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the file by gene pair sections
    gene_pair_sections = re.split(r'# Interaction Pathways: ([A-Za-z0-9]+) → ([A-Za-z0-9]+)', content)[1:]  # Skip the first element which is before any match
    
    results = []
    
    # Process each gene pair section
    for i in range(0, len(gene_pair_sections), 3):  # Process in triplets (source, target, content)
        if i + 2 < len(gene_pair_sections):
            source_gene = gene_pair_sections[i]
            target_gene = gene_pair_sections[i+1]
            section_content = gene_pair_sections[i+2]
            
            # Extract pathway summaries
            pathway_blocks = re.split(r'### Path \d+: ([A-Za-z-]+)', section_content)[1:]  # Skip the first element
            
            pathways = []
            for j in range(0, len(pathway_blocks), 2):  # Process in pairs (classification, content)
                if j + 1 < len(pathway_blocks):
                    classification = pathway_blocks[j]
                    pathway_content = pathway_blocks[j+1]
                    
                    # Extract summary
                    summary_match = re.search(r'\*\*Summary\*\*: (.*?)\n', pathway_content)
                    summary = summary_match.group(1) if summary_match else ""
                    
                    # Extract overall probability
                    prob_match = re.search(r'\*\*Overall Probability\*\*: ([\d.]+)', pathway_content)
                    probability = float(prob_match.group(1)) if prob_match else 0.0
                    
                    # Extract support ratio
                    support_match = re.search(r'\*\*Support Ratio\*\*: ([\d.]+) \((\d+)/(\d+)', pathway_content)
                    support_ratio = float(support_match.group(1)) if support_match else 0.0
                    supported_edges = int(support_match.group(2)) if support_match else 0
                    total_edges = int(support_match.group(3)) if support_match else 0
                    
                    # Extract interactions
                    interactions = []
                    interaction_blocks = re.split(r'\*\*Interaction \d+\*\*: ([A-Za-z0-9]+) → ([A-Za-z0-9]+)', pathway_content)[1:]
                    
                    for k in range(0, len(interaction_blocks), 3):  # Process in triplets (source, target, content)
                        if k + 2 < len(interaction_blocks):
                            int_source = interaction_blocks[k]
                            int_target = interaction_blocks[k+1]
                            int_content = interaction_blocks[k+2]
                            
                            # Extract mechanism
                            mech_match = re.search(r'\*\*Mechanism\*\*: (.*?)\n', int_content)
                            mechanism = mech_match.group(1) if mech_match else ""
                            
                            # Extract probability
                            int_prob_match = re.search(r'\*\*Probability\*\*: ([\d.]+)', int_content)
                            int_probability = float(int_prob_match.group(1)) if int_prob_match else 0.0
                            
                            # Extract evidence
                            evidence_details = ""
                            if include_evidence:
                                evidence_match = re.search(r'\*\*Evidence\*\* \([^)]+\): (.*?)(?:\n\n|$)', int_content, re.DOTALL)
                                evidence_details = evidence_match.group(1).strip() if evidence_match else ""
                            
                            interaction = {
                                "source": int_source,
                                "target": int_target,
                                "mechanism": mechanism,
                                "probability": int_probability
                            }
                            
                            if include_evidence:
                                interaction["evidence"] = evidence_details
                                
                            interactions.append(interaction)
                    
                    pathway = {
                        "summary": summary,
                        "classification": classification,
                        "overall_probability": probability,
                        "support_ratio": support_ratio,
                        "supported_edges": supported_edges,
                        "total_edges": total_edges,
                        "interactions": interactions
                    }
                    
                    pathways.append(pathway)
            
            # Limit the number of pathways if specified
            if path_limit is not None and path_limit > 0:
                pathways = pathways[:path_limit]
                
            gene_pair_result = {
                "source_gene": source_gene,
                "target_gene": target_gene,
                "pathways": pathways
            }
            
            results.append(gene_pair_result)
    
    return results


def export_to_json(pathways_data: List[Dict[str, Any]], output_file: str) -> None:
    """Export pathway data to JSON format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pathways_data, f, indent=2)
    print(f"Exported pathway data to {output_file}")


def export_to_csv(pathways_data: List[Dict[str, Any]], output_file: str, delimiter: str = ',') -> None:
    """Export pathway data to CSV or TSV format."""
    # Flatten the data for tabular format
    rows = []
    for gene_pair in pathways_data:
        source = gene_pair["source_gene"]
        target = gene_pair["target_gene"]
        
        for path_idx, pathway in enumerate(gene_pair["pathways"], 1):
            for interaction in pathway["interactions"]:
                row = {
                    "source_gene": source,
                    "target_gene": target,
                    "pathway_number": path_idx,
                    "pathway_summary": pathway["summary"],
                    "pathway_classification": pathway["classification"],
                    "pathway_probability": pathway["overall_probability"],
                    "support_ratio": pathway["support_ratio"],
                    "interaction_source": interaction["source"],
                    "interaction_target": interaction["target"],
                    "mechanism": interaction["mechanism"],
                    "interaction_probability": interaction["probability"]
                }
                
                if "evidence" in interaction:
                    row["evidence"] = interaction["evidence"]
                    
                rows.append(row)
    
    if rows:
        fieldnames = rows[0].keys()
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported pathway data to {output_file}")
    else:
        print("No data to export")


def export_to_markdown(pathways_data: List[Dict[str, Any]], output_file: str) -> None:
    """Export pathway data to markdown format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Gene Pathway Interactions Summary\n\n")
        
        for gene_pair in pathways_data:
            source = gene_pair["source_gene"]
            target = gene_pair["target_gene"]
            
            f.write(f"## {source} → {target}\n\n")
            f.write("### Pathways Summary\n\n")
            f.write("| # | Classification | Support | Probability | Summary |\n")
            f.write("|---|---------------|---------|-------------|--------|\n")
            
            for path_idx, pathway in enumerate(gene_pair["pathways"], 1):
                f.write(f"| {path_idx} | {pathway['classification']} | {pathway['support_ratio']:.2f} ({pathway['supported_edges']}/{pathway['total_edges']}) | {pathway['overall_probability']:.2f} | {pathway['summary']} |\n")
            
            f.write("\n### Detailed Interactions\n\n")
            
            for path_idx, pathway in enumerate(gene_pair["pathways"], 1):
                f.write(f"**Path {path_idx}**: {pathway['summary']}\n\n")
                
                for int_idx, interaction in enumerate(pathway["interactions"], 1):
                    f.write(f"- {interaction['source']} → {interaction['target']} ({interaction['mechanism']}, P={interaction['probability']:.2f})\n")
                    if "evidence" in interaction:
                        f.write(f"  - Evidence: {interaction['evidence']}\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        print(f"Exported pathway data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract pathway information from a markdown file")
    parser.add_argument("input_file", help="The input markdown file to extract pathways from")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: uses input filename with new extension)")
    parser.add_argument("--format", choices=["json", "csv", "tsv", "md"], default="json",
                        help="Output format: json, csv, tsv, or md (default: json)")
    parser.add_argument("--include-evidence", action="store_true",
                        help="Include evidence details in the output")
    parser.add_argument("--path-limit", type=int, default=None,
                        help="Limit the number of pathways per gene pair")
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Generate default output filename if not provided
    if args.output is None:
        input_base = os.path.splitext(args.input_file)[0]
        if args.format == "json":
            args.output = f"{input_base}_pathways.json"
        elif args.format == "csv":
            args.output = f"{input_base}_pathways.csv"
        elif args.format == "tsv":
            args.output = f"{input_base}_pathways.tsv"
        elif args.format == "md":
            args.output = f"{input_base}_pathways.md"
    
    print(f"Processing {args.input_file}...")
    pathways = extract_pathways(args.input_file, args.include_evidence, args.path_limit)
    
    # Export data in the requested format
    if args.format == "json":
        export_to_json(pathways, args.output)
    elif args.format == "csv":
        export_to_csv(pathways, args.output, delimiter=',')
    elif args.format == "tsv":
        export_to_csv(pathways, args.output, delimiter='\t')
    elif args.format == "md":
        export_to_markdown(pathways, args.output)
    else:
        print(f"Unsupported format: {args.format}")


if __name__ == "__main__":
    main()