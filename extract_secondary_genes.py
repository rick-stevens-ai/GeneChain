#!/usr/bin/env python3
"""
Extract Secondary Gene Analysis from Pathway Files

This script analyzes *-TF_pathways.json files to identify:
1. The primary gene pairs (direct source and target genes)
2. Secondary (intermediate) genes that appear in the pathways
3. Frequency analysis of secondary genes across all pathways
4. Tentative interactions that use words like "may" in their descriptions

It produces a summary of how often each secondary gene appears in pathways,
helping to identify key mediator genes in the network.

Usage:
------
$ python extract_secondary_genes.py [--input-pattern INPUT_PATTERN] [--output OUTPUT_FILE]

Arguments:
  --input-pattern INPUT_PATTERN    Glob pattern for input files (default: *-TF_pathways.json)
  --output OUTPUT_FILE             Output JSON file (default: secondary_genes_analysis.json)
  --min-occurrences MIN            Only include interactions/genes with at least MIN occurrences (default: 1)
  --sort-by {occurrences,name}     Sort results by occurrence count or gene name (default: occurrences)
  --show-tentative                 Identify and show tentative interactions (containing "may", "might", "possibly", etc.)
  --only-tentative                 Only display tentative interactions (can be combined with min-occurrences)
  --list-tentative-pairs           Just list gene pairs that have tentative interactions (simple output format)

When --only-tentative is used with --min-occurrences:
  The script counts how many times each specific interaction (same source, target, and mechanism) appears
  across all files, and only shows interactions that appear at least MIN times.
"""

import argparse
import glob
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set, Tuple

def load_pathway_files(pattern: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load all pathway files matching the glob pattern."""
    file_paths = glob.glob(pattern)
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(file_paths)} files: {', '.join(file_paths)}")
    
    files_data = {}
    for file_path in file_paths:
        file_id = os.path.basename(file_path).split('-')[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            files_data[file_id] = data
            print(f"Loaded data from {file_path}")
    
    return files_data

def is_tentative_interaction(text: str) -> bool:
    """
    Check if an interaction description contains tentative language.
    Returns True if the text contains words like "may", "might", "possibly", etc.
    """
    tentative_words = [
        "may", "might", "could", "potentially", "possibly", "perhaps", 
        "suggest", "putative", "hypothesized", "predicted", "probable",
        "likely", "proposed", "presumed", "postulated"
    ]
    
    if not text:
        return False
        
    text_lower = text.lower()
    return any(word in text_lower for word in tentative_words)
    
def analyze_secondary_genes(files_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze the pathway data to extract primary pairs and secondary genes.
    Returns a dictionary with analysis results.
    """
    # Track primary gene pairs
    primary_pairs = set()
    
    # Track secondary genes and their occurrences
    secondary_genes = Counter()
    
    # Track which primary pairs involve each secondary gene
    gene_to_primary_pairs = defaultdict(set)
    
    # Track pathways containing each secondary gene
    gene_to_pathways = defaultdict(list)
    
    # Track the role of each gene (primary source, primary target, secondary)
    gene_roles = defaultdict(set)
    
    # Track tentative interactions
    tentative_interactions = []
    
    # Process each file
    for file_id, gene_pairs in files_data.items():
        # Process each gene pair
        for gene_pair in gene_pairs:
            source_gene = gene_pair.get("source_gene", "")
            target_gene = gene_pair.get("target_gene", "")
            
            if not source_gene or not target_gene:
                continue
                
            # Add to primary pairs
            primary_pair = (source_gene, target_gene)
            primary_pairs.add(primary_pair)
            
            # Track primary gene roles
            gene_roles[source_gene].add("primary_source")
            gene_roles[target_gene].add("primary_target")
            
            # Process each pathway
            for pathway in gene_pair.get("pathways", []):
                # Check for tentative language in pathway summary
                pathway_summary = pathway.get("summary", "")
                pathway_is_tentative = is_tentative_interaction(pathway_summary)
                
                # Process interactions to find secondary genes
                interactions = pathway.get("interactions", [])
                
                # Skip pathways with no interactions
                if not interactions:
                    continue
                
                # Track all genes in this pathway
                pathway_genes = set()
                pathway_tentative_interactions = []
                
                # First pass: collect all genes in the pathway and check for tentative interactions
                for interaction in interactions:
                    int_source = interaction.get("source", "")
                    int_target = interaction.get("target", "")
                    mechanism = interaction.get("mechanism", "")
                    
                    if int_source:
                        pathway_genes.add(int_source)
                    if int_target:
                        pathway_genes.add(int_target)
                        
                    # Check for tentative language in the interaction mechanism
                    is_tentative = is_tentative_interaction(mechanism)
                    
                    # Add to tentative interactions if either the pathway or this specific interaction is tentative
                    if pathway_is_tentative or is_tentative:
                        tentative_int = {
                            "file": file_id,
                            "primary_source": source_gene,
                            "primary_target": target_gene,
                            "interaction_source": int_source,
                            "interaction_target": int_target,
                            "mechanism": mechanism,
                            "pathway_summary": pathway_summary,
                            "probability": interaction.get("probability", 0.0),
                            "tentative_in": "pathway" if pathway_is_tentative else "mechanism"
                        }
                        tentative_interactions.append(tentative_int)
                        pathway_tentative_interactions.append(tentative_int)
                
                # Second pass: identify secondary genes
                # A secondary gene is any gene in the pathway that is not the primary source or target
                secondary_in_pathway = pathway_genes - {source_gene, target_gene}
                
                # Update secondary gene counts
                for gene in secondary_in_pathway:
                    secondary_genes[gene] += 1
                    gene_to_primary_pairs[gene].add(primary_pair)
                    gene_roles[gene].add("secondary")
                    
                    # Store pathway information for this secondary gene
                    pathway_info = {
                        "file": file_id,
                        "primary_source": source_gene,
                        "primary_target": target_gene,
                        "summary": pathway_summary,
                        "probability": pathway.get("overall_probability", 0.0),
                        "is_tentative": pathway_is_tentative,
                        "tentative_interactions": [
                            t for t in pathway_tentative_interactions 
                            if t["interaction_source"] == gene or t["interaction_target"] == gene
                        ]
                    }
                    gene_to_pathways[gene].append(pathway_info)
    
    # Prepare the final analysis result
    result = {
        "primary_pairs": [{"source": s, "target": t} for s, t in primary_pairs],
        "primary_pairs_count": len(primary_pairs),
        "secondary_genes": [
            {
                "gene": gene,
                "occurrences": count,
                "primary_pairs": [
                    {"source": s, "target": t}
                    for s, t in sorted(gene_to_primary_pairs[gene])
                ],
                "primary_pairs_count": len(gene_to_primary_pairs[gene]),
                "pathways": gene_to_pathways[gene],
                "roles": sorted(gene_roles[gene])
            }
            for gene, count in secondary_genes.items()
        ],
        "all_genes": {
            gene: {"roles": sorted(roles)} 
            for gene, roles in gene_roles.items()
        },
        "tentative_interactions": tentative_interactions,
        "tentative_interactions_count": len(tentative_interactions)
    }
    
    return result

def print_secondary_genes_summary(analysis: Dict[str, Any], min_occurrences: int = 1, 
                                sort_by: str = "occurrences", show_tentative: bool = False,
                                only_tentative: bool = False) -> None:
    """Print a summary of secondary genes to the console."""
    # If only showing tentative interactions, skip the regular summary
    if not only_tentative:
        print("\n===== SECONDARY GENES ANALYSIS =====\n")
        print(f"Total Primary Gene Pairs: {analysis['primary_pairs_count']}")
        
        # Get and potentially filter secondary genes
        secondary_genes = analysis.get("secondary_genes", [])
        if min_occurrences > 1:
            secondary_genes = [g for g in secondary_genes if g["occurrences"] >= min_occurrences]
        
        print(f"Total Secondary Genes: {len(secondary_genes)}")
        
        # Sort secondary genes
        if sort_by == "occurrences":
            secondary_genes = sorted(secondary_genes, key=lambda g: (-g["occurrences"], g["gene"]))
        else:  # sort by name
            secondary_genes = sorted(secondary_genes, key=lambda g: g["gene"])
        
        # Print table header
        print("\nSecondary Genes by Occurrence:")
        print(f"{'Gene':<15} {'Occurrences':<12} {'Primary Pairs':<15} {'Roles'}")
        print("-" * 70)
        
        # Print each secondary gene
        for gene_info in secondary_genes:
            gene = gene_info["gene"]
            occurrences = gene_info["occurrences"]
            primary_pairs_count = gene_info["primary_pairs_count"]
            roles = ", ".join(gene_info["roles"])
            
            print(f"{gene:<15} {occurrences:<12} {primary_pairs_count:<15} {roles}")
        
        # Print most common pathway connections
        print("\n\n===== DETAILED SECONDARY GENE ANALYSIS =====\n")
        
        for gene_info in secondary_genes[:10]:  # Limit to top 10 for brevity
            gene = gene_info["gene"]
            occurrences = gene_info["occurrences"]
            
            print(f"\n{gene} (Occurs in {occurrences} pathways):")
            
            # Group by primary pair
            pair_pathways = defaultdict(list)
            for pathway in gene_info["pathways"]:
                pair_key = f"{pathway['primary_source']} → {pathway['primary_target']}"
                pair_pathways[pair_key].append(pathway)
            
            # Print pathways grouped by primary pair
            for pair, pathways in sorted(pair_pathways.items()):
                print(f"  {pair} ({len(pathways)} pathways):")
                for i, pathway in enumerate(pathways, 1):
                    file = pathway["file"]
                    summary = pathway["summary"]
                    prob = pathway["probability"]
                    tentative_marker = " [TENTATIVE]" if pathway.get("is_tentative", False) else ""
                    print(f"    {i}. File {file}: {summary} (P={prob}){tentative_marker}")
    
    # Print tentative interactions if requested
    if show_tentative:
        tentative_interactions = analysis.get("tentative_interactions", [])
        
        # Count occurrences of each specific interaction (source-target-mechanism combination)
        interaction_counts = Counter()
        for interaction in tentative_interactions:
            # Create a key that uniquely identifies this interaction
            interaction_key = (
                interaction["interaction_source"],
                interaction["interaction_target"],
                interaction["mechanism"]
            )
            interaction_counts[interaction_key] += 1
        
        # Create an enhanced list with occurrence counts and filter by min_occurrences
        enhanced_interactions = []
        seen_keys = set()
        
        for interaction in tentative_interactions:
            interaction_key = (
                interaction["interaction_source"],
                interaction["interaction_target"],
                interaction["mechanism"]
            )
            
            # Skip if we've already processed this exact interaction
            if interaction_key in seen_keys:
                continue
                
            # Mark this interaction as seen
            seen_keys.add(interaction_key)
            
            # Get occurrence count
            count = interaction_counts[interaction_key]
            
            # Only include if it meets the min_occurrences threshold
            if count >= min_occurrences:
                # Create a copy with the occurrence count
                enhanced = dict(interaction)
                enhanced["occurrences"] = count
                enhanced_interactions.append(enhanced)
        
        # Replace the original list with the enhanced and filtered list
        tentative_interactions = enhanced_interactions
        
        # Print header for tentative interactions section
        if only_tentative:
            print("\n===== TENTATIVE INTERACTIONS ANALYSIS =====")
            print(f"\nShowing interactions that:")
            print(f"  1. Contain tentative language ('may', 'might', 'could', etc.)")
            print(f"  2. Appear at least {min_occurrences} time(s) with the same source, target, and mechanism")
        
        print(f"\n===== TENTATIVE INTERACTIONS ({len(tentative_interactions)}) =====\n")
        
        if not tentative_interactions:
            print("No tentative interactions found with the current filtering criteria.")
            return
        
        # Sort tentative interactions by occurrence count (descending)
        tentative_interactions = sorted(
            tentative_interactions,
            key=lambda x: (-x["occurrences"], x["interaction_source"], x["interaction_target"])
        )
        
        # Group tentative interactions by file
        by_file = defaultdict(list)
        for interaction in tentative_interactions:
            by_file[interaction["file"]].append(interaction)
        
        # Print tentative interactions grouped by file
        for file_id, interactions in sorted(by_file.items()):
            print(f"\nFile {file_id} ({len(interactions)} tentative interactions):")
            
            # Group by primary pair
            by_pair = defaultdict(list)
            for interaction in interactions:
                pair = f"{interaction['primary_source']} → {interaction['primary_target']}"
                by_pair[pair].append(interaction)
            
            # Print interactions grouped by primary pair
            for pair, pair_interactions in sorted(by_pair.items()):
                print(f"  {pair}:")
                
                # Print each tentative interaction
                for i, interaction in enumerate(pair_interactions, 1):
                    source = interaction["interaction_source"]
                    target = interaction["interaction_target"]
                    mechanism = interaction["mechanism"]
                    pathway_summary = interaction["pathway_summary"]
                    tentative_in = interaction["tentative_in"]
                    probability = interaction["probability"]
                    occurrences = interaction["occurrences"]
                    
                    print(f"    {i}. {source} → {target}: {mechanism} (P={probability}, Occurrences: {occurrences})")
                    print(f"       Pathway: {pathway_summary}")
                    print(f"       Tentative in: {tentative_in}")
                    print(f"       -")
        
        print("\nNote: Tentative interactions contain words like 'may', 'might', 'could', etc.")

def list_tentative_pairs(analysis: Dict[str, Any], min_occurrences: int = 1) -> None:
    """
    Print a simple list of gene pairs that have tentative interactions.
    This provides a concise overview of which gene pairs involve tentative language.
    """
    tentative_interactions = analysis.get("tentative_interactions", [])
    
    if not tentative_interactions:
        print("No tentative interactions found.")
        return
    
    # Extract unique (source, target) gene pairs from tentative interactions
    tentative_pairs = set()
    # Also track interaction keys to count occurrences
    interaction_counts = Counter()
    
    for interaction in tentative_interactions:
        # Extract the gene pair from the interaction
        source = interaction["interaction_source"]
        target = interaction["interaction_target"]
        
        # Add to the set of tentative pairs
        tentative_pairs.add((source, target))
        
        # Count occurrences of each specific interaction
        interaction_key = (source, target, interaction["mechanism"])
        interaction_counts[interaction_key] += 1
    
    # For each pair, count how many times it appears in interactions meeting min_occurrences
    pair_counts = Counter()
    
    for interaction in tentative_interactions:
        source = interaction["interaction_source"]
        target = interaction["interaction_target"]
        key = (source, target, interaction["mechanism"])
        
        if interaction_counts[key] >= min_occurrences:
            pair_counts[(source, target)] += 1
    
    # Filter to pairs with interactions meeting the threshold
    valid_pairs = [(s, t) for (s, t) in tentative_pairs if pair_counts[(s, t)] > 0]
    
    # Sort by occurrence count first, then alphabetically
    sorted_pairs = sorted(valid_pairs, key=lambda p: (-pair_counts[p], p[0], p[1]))
    
    print(f"\n===== GENE PAIRS WITH TENTATIVE INTERACTIONS ({len(sorted_pairs)}) =====\n")
    print(f"Showing pairs where at least one tentative interaction occurs {min_occurrences}+ times\n")
    
    print(f"{'Source Gene':<15} {'Target Gene':<15} {'Tentative Interactions'}")
    print("-" * 60)
    
    for source, target in sorted_pairs:
        count = pair_counts[(source, target)]
        print(f"{source:<15} {target:<15} {count}")
    
    print("\nNote: Tentative interactions contain words like 'may', 'might', 'could', etc.")

def main():
    parser = argparse.ArgumentParser(description="Extract and analyze secondary genes from pathway files")
    parser.add_argument("--input-pattern", default="*-TF_pathways.json",
                        help="Glob pattern for input files (default: *-TF_pathways.json)")
    parser.add_argument("--output", default="secondary_genes_analysis.json",
                        help="Output JSON file (default: secondary_genes_analysis.json)")
    parser.add_argument("--min-occurrences", type=int, default=1,
                        help="Only include genes with at least MIN occurrences (default: 1)")
    parser.add_argument("--sort-by", choices=["occurrences", "name"], default="occurrences",
                        help="Sort results by occurrence count or gene name (default: occurrences)")
    parser.add_argument("--show-tentative", action="store_true",
                        help="Identify and show tentative interactions (containing 'may', 'might', 'possibly', etc.)")
    parser.add_argument("--only-tentative", action="store_true", 
                        help="Only display tentative interactions (can be combined with min-occurrences)")
    parser.add_argument("--list-tentative-pairs", action="store_true",
                        help="Just list gene pairs that have tentative interactions (simple output format)")
    args = parser.parse_args()
    
    try:
        # Load all pathway files
        files_data = load_pathway_files(args.input_pattern)
        
        # Analyze secondary genes
        analysis = analyze_secondary_genes(files_data)
        
        # Handle the different output modes
        if args.list_tentative_pairs:
            # Just list gene pairs with tentative interactions
            list_tentative_pairs(analysis, args.min_occurrences)
        else:
            # When --only-tentative is used, always show tentative interactions
            if args.only_tentative:
                args.show_tentative = True
            
            # Print the standard summary
            print_secondary_genes_summary(analysis, args.min_occurrences, args.sort_by, 
                                          args.show_tentative, args.only_tentative)
        
        # Save analysis to file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nFull analysis saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()