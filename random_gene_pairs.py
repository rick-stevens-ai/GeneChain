#!/usr/bin/env python3
"""
Random Gene Pair Generator
=========================

This script takes a CSV or TSV file containing gene names and generates random
pairs of genes (without replacement) for further analysis.

Usage:
------
python random_gene_pairs.py --input-file rad_genes_132.tsv --num-pairs 10

Options:
--------
--input-file  : Path to the input CSV or TSV file containing gene names
--num-pairs   : Number of random pairs to generate (default: 5)
--gene-column : Name or index of the column containing gene names (default: 0 or 'Gene')
--delimiter   : Delimiter for the output file (default: tab)
--output-file : Path to the output file (default: pairs.txt)
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List, Set, Tuple


def load_genes(file_path: str, gene_column: str = None) -> List[str]:
    """
    Load gene names from a CSV or TSV file.
    
    Args:
        file_path: Path to the input file
        gene_column: Column name or index containing gene names (default: try 'Gene' or index 0)
        
    Returns:
        List of gene names
    """
    # Determine file extension to guess the delimiter
    ext = os.path.splitext(file_path)[1].lower()
    default_delimiter = ',' if ext == '.csv' else '\t'
    
    try:
        with open(file_path, 'r') as f:
            # Try to detect if there's a header
            sample = f.readline()
            f.seek(0)  # Reset file pointer
            
            # Check if sample line has alphabetic characters in first cell (likely a header)
            has_header = any(c.isalpha() for c in sample.split(default_delimiter)[0])
            
            if has_header:
                reader = csv.DictReader(f, delimiter=default_delimiter)
                fieldnames = reader.fieldnames
                
                # Determine which column to use for genes
                col_to_use = None
                if gene_column is not None:
                    if gene_column.isdigit():
                        # Convert to column name if it's a number
                        idx = int(gene_column)
                        if idx < len(fieldnames):
                            col_to_use = fieldnames[idx]
                    elif gene_column in fieldnames:
                        col_to_use = gene_column
                
                # Fallback to 'Gene' or first column
                if col_to_use is None:
                    if 'Gene' in fieldnames:
                        col_to_use = 'Gene'
                    else:
                        col_to_use = fieldnames[0]
                
                # Extract genes from the column
                genes = [row[col_to_use].strip() for row in reader if row[col_to_use].strip()]
            else:
                # No header, use column index
                reader = csv.reader(f, delimiter=default_delimiter)
                col_idx = 0
                if gene_column is not None and gene_column.isdigit():
                    col_idx = int(gene_column)
                
                genes = []
                for row in reader:
                    if len(row) > col_idx and row[col_idx].strip():
                        genes.append(row[col_idx].strip())
                        
        # Remove any duplicates while preserving order
        seen: Set[str] = set()
        unique_genes = [g for g in genes if not (g in seen or seen.add(g))]
        
        return unique_genes
        
    except Exception as e:
        print(f"[ERROR] Failed to load genes from {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_random_pairs(genes: List[str], num_pairs: int) -> List[Tuple[str, str]]:
    """
    Generate random unique pairs of genes without replacement.
    
    Args:
        genes: List of gene names
        num_pairs: Number of pairs to generate
        
    Returns:
        List of gene pairs
    """
    if len(genes) < 2:
        print("[ERROR] Need at least 2 genes to create pairs", file=sys.stderr)
        sys.exit(1)
    
    # Calculate maximum possible number of pairs
    max_pairs = (len(genes) * (len(genes) - 1)) // 2
    
    if num_pairs > max_pairs:
        print(f"[WARN] Requested {num_pairs} pairs but only {max_pairs} are possible. Using maximum.", file=sys.stderr)
        num_pairs = max_pairs
    
    # Generate all possible pairs
    all_pairs = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            all_pairs.append((genes[i], genes[j]))
    
    # Randomly select the requested number of pairs
    return random.sample(all_pairs, num_pairs)


def write_pairs(pairs: List[Tuple[str, str]], file_path: str, delimiter: str) -> None:
    """
    Write gene pairs to the output file.
    
    Args:
        pairs: List of gene pairs
        file_path: Path to the output file
        delimiter: Delimiter to use between genes
    """
    try:
        with open(file_path, 'w') as f:
            for gene1, gene2 in pairs:
                f.write(f"{gene1}{delimiter}{gene2}\n")
        print(f"[INFO] Successfully wrote {len(pairs)} gene pairs to {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write pairs to {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random gene pairs from a CSV/TSV file")
    parser.add_argument("--input-file", required=True, help="Path to the input CSV or TSV file")
    parser.add_argument("--num-pairs", type=int, default=5, help="Number of random pairs to generate (default: 5)")
    parser.add_argument("--gene-column", default=None, help="Name or index of the column containing gene names (default: tries 'Gene' or first column)")
    parser.add_argument("--delimiter", default="\t", help="Delimiter for the output file (default: tab)")
    parser.add_argument("--output-file", default="pairs.txt", help="Path to the output file (default: pairs.txt)")
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Load genes from the input file
    genes = load_genes(args.input_file, args.gene_column)
    print(f"[INFO] Loaded {len(genes)} unique genes from {args.input_file}")
    
    # Generate random pairs
    pairs = generate_random_pairs(genes, args.num_pairs)
    
    # Write pairs to the output file
    write_pairs(pairs, args.output_file, args.delimiter)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()