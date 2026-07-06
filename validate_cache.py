#!/usr/bin/env python3
"""
INTERACTION_CACHE Validation Analyzer
=====================================

This script validates all gene interactions stored in the INTERACTION_CACHE directory
against known biological databases to classify them as validated (known) vs. 
conjectural (novel/unvalidated) interactions.

Key Features:
- Batch validates all interactions in INTERACTION_CACHE
- Classifies interactions as Validated, Conjectural, or Novel
- Generates comprehensive reports with confidence scoring
- Identifies high-priority novel interactions for experimental validation
- Creates summary statistics for research planning

Usage:
------
$ python validate_cache.py
$ python validate_cache.py --cache-dir INTERACTION_CACHE --output cache_validation_report
$ python validate_cache.py --summary-only --min-confidence 0.7

Arguments:
  --cache-dir DIR        Path to INTERACTION_CACHE directory (default: INTERACTION_CACHE)
  --output DIR           Output directory for validation results (default: cache_validation_TIMESTAMP)
  --summary-only         Generate only summary statistics, not detailed reports
  --min-confidence FLOAT Minimum confidence threshold for validation (default: 0.4)
  --max-interactions INT Maximum number of interactions to validate (default: unlimited)
  --novel-priority       Focus analysis on identifying novel interactions
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import logging

# Import our validation system
from validate_interactions import DatabaseValidator, load_interaction_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CacheValidationAnalyzer:
    """Analyzes and validates all interactions in the INTERACTION_CACHE."""
    
    def __init__(self, cache_dir: str = "INTERACTION_CACHE", 
                 validation_cache_dir: str = "cache_validation_db_cache"):
        self.cache_dir = Path(cache_dir)
        self.validator = DatabaseValidator(cache_dir=validation_cache_dir)
        
    def discover_interaction_files(self) -> List[Path]:
        """Find all interaction JSON files in the cache directory."""
        if not self.cache_dir.exists():
            logger.error(f"Cache directory not found: {self.cache_dir}")
            return []
            
        # Look for interaction files
        patterns = [
            "*_interactions.json",
            "network_*_*_interactions.json"
        ]
        
        interaction_files = []
        for pattern in patterns:
            files = list(self.cache_dir.glob(pattern))
            interaction_files.extend(files)
        
        # Remove duplicates
        interaction_files = list(set(interaction_files))
        logger.info(f"Found {len(interaction_files)} interaction files in {self.cache_dir}")
        
        return interaction_files
    
    def extract_gene_pairs_from_cache(self, interaction_files: List[Path]) -> Dict[str, Dict[str, Any]]:
        """Extract all unique gene pairs and their interactions from cache files."""
        gene_pair_data = {}
        
        for file_path in interaction_files:
            logger.info(f"Processing: {file_path.name}")
            
            # Load interaction data
            interactions = load_interaction_data(str(file_path))
            
            # Extract gene pairs
            for interaction in interactions:
                gene_a = interaction["gene_a"]
                gene_b = interaction["gene_b"]
                
                # Create a unique key for this gene pair
                pair_key = tuple(sorted([gene_a, gene_b]))
                
                if pair_key not in gene_pair_data:
                    gene_pair_data[pair_key] = {
                        "gene_a": pair_key[0],
                        "gene_b": pair_key[1],
                        "interactions": [],
                        "source_files": [],
                        "ai_predictions": []
                    }
                
                # Add this interaction
                gene_pair_data[pair_key]["interactions"].append(interaction)
                gene_pair_data[pair_key]["source_files"].append(str(file_path))
                
                # Store AI prediction details
                ai_prediction = {
                    "mechanism": interaction.get("mechanism", ""),
                    "probability": interaction.get("probability", 0),
                    "evidence": interaction.get("evidence", ""),
                    "source_file": str(file_path)
                }
                gene_pair_data[pair_key]["ai_predictions"].append(ai_prediction)
        
        logger.info(f"Extracted {len(gene_pair_data)} unique gene pairs from cache")
        return gene_pair_data
    
    def validate_gene_pairs(self, gene_pair_data: Dict[str, Dict[str, Any]], 
                          max_interactions: int = None) -> Dict[str, Any]:
        """Validate all gene pairs against databases."""
        validation_results = {}
        pairs_to_process = list(gene_pair_data.items())
        
        if max_interactions:
            pairs_to_process = pairs_to_process[:max_interactions]
            logger.info(f"Limiting validation to {max_interactions} gene pairs")
        
        for i, (pair_key, pair_data) in enumerate(pairs_to_process, 1):
            gene_a = pair_data["gene_a"]
            gene_b = pair_data["gene_b"]
            
            logger.info(f"Validating {i}/{len(pairs_to_process)}: {gene_a} - {gene_b}")
            
            # Perform comprehensive validation
            validation_result = self.validator.comprehensive_validation(gene_a, gene_b)
            
            # Add cache-specific information
            validation_result["cache_info"] = {
                "interaction_count": len(pair_data["interactions"]),
                "source_files": pair_data["source_files"],
                "ai_predictions": pair_data["ai_predictions"]
            }
            
            validation_results[pair_key] = validation_result
            
            # Brief progress update
            validation_score = validation_result["validation_summary"]["validation_score"]
            confidence = validation_result["validation_summary"]["confidence_level"]
            novel = validation_result["novel_interaction"]
            
            status = "NOVEL" if novel else f"KNOWN ({confidence})"
            logger.info(f"  Result: {status} (validation score: {validation_score:.2f})")
        
        return validation_results
    
    def analyze_validation_results(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze validation results and generate insights."""
        analysis = {
            "summary_statistics": {
                "total_gene_pairs": len(validation_results),
                "validated_pairs": 0,
                "novel_pairs": 0,
                "high_confidence_pairs": 0,
                "medium_confidence_pairs": 0,
                "low_confidence_pairs": 0
            },
            "confidence_distribution": Counter(),
            "database_coverage": Counter(),
            "validated_interactions": [],
            "novel_interactions": [],
            "high_priority_novel": [],
            "low_confidence_predictions": []
        }
        
        for pair_key, result in validation_results.items():
            gene_a, gene_b = pair_key
            validation_summary = result["validation_summary"]
            cache_info = result["cache_info"]
            
            # Update statistics
            validation_score = validation_summary["validation_score"]
            confidence_level = validation_summary["confidence_level"]
            is_novel = result["novel_interaction"]
            
            analysis["confidence_distribution"][confidence_level] += 1
            
            if validation_score > 0:
                analysis["summary_statistics"]["validated_pairs"] += 1
                
                # Categorize by confidence
                if confidence_level == "High":
                    analysis["summary_statistics"]["high_confidence_pairs"] += 1
                elif confidence_level == "Medium":
                    analysis["summary_statistics"]["medium_confidence_pairs"] += 1
                else:
                    analysis["summary_statistics"]["low_confidence_pairs"] += 1
                
                # Add to validated interactions
                validated_entry = {
                    "gene_pair": (gene_a, gene_b),
                    "confidence_level": confidence_level,
                    "validation_score": validation_score,
                    "databases_validated": validation_summary["validated_databases"],
                    "ai_predictions": cache_info["ai_predictions"],
                    "source_files": cache_info["source_files"]
                }
                analysis["validated_interactions"].append(validated_entry)
                
            else:
                analysis["summary_statistics"]["novel_pairs"] += 1
                
                # Add to novel interactions
                novel_entry = {
                    "gene_pair": (gene_a, gene_b),
                    "ai_predictions": cache_info["ai_predictions"],
                    "source_files": cache_info["source_files"],
                    "evidence_strength": result["evidence_strength"]
                }
                analysis["novel_interactions"].append(novel_entry)
                
                # Check if high priority for experimental validation
                ai_probs = [pred["probability"] for pred in cache_info["ai_predictions"]]
                max_ai_prob = max(ai_probs) if ai_probs else 0
                
                if max_ai_prob >= 0.7:  # High AI confidence but no database validation
                    novel_entry["max_ai_probability"] = max_ai_prob
                    novel_entry["priority_score"] = max_ai_prob  # Simple priority scoring
                    analysis["high_priority_novel"].append(novel_entry)
            
            # Track database coverage
            for db_name, db_result in result["database_results"].items():
                if db_result.get("validated", False):
                    analysis["database_coverage"][db_name] += 1
        
        # Sort lists by priority/confidence
        analysis["validated_interactions"].sort(
            key=lambda x: x["validation_score"], reverse=True
        )
        analysis["novel_interactions"].sort(
            key=lambda x: max([p["probability"] for p in x["ai_predictions"]] + [0]), 
            reverse=True
        )
        analysis["high_priority_novel"].sort(
            key=lambda x: x["priority_score"], reverse=True
        )
        
        return analysis
    
    def generate_reports(self, analysis: Dict[str, Any], output_dir: Path):
        """Generate comprehensive validation reports."""
        output_dir.mkdir(exist_ok=True)
        
        # 1. Summary Statistics Report
        summary_file = output_dir / "validation_summary.txt"
        with open(summary_file, 'w') as f:
            stats = analysis["summary_statistics"]
            f.write("INTERACTION_CACHE Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total gene pairs analyzed: {stats['total_gene_pairs']}\n")
            f.write(f"  Validated (known) pairs: {stats['validated_pairs']}\n")
            f.write(f"  Novel (unvalidated) pairs: {stats['novel_pairs']}\n\n")
            
            if stats['total_gene_pairs'] > 0:
                val_rate = (stats['validated_pairs'] / stats['total_gene_pairs']) * 100
                novel_rate = (stats['novel_pairs'] / stats['total_gene_pairs']) * 100
                f.write(f"  Validation rate: {val_rate:.1f}%\n")
                f.write(f"  Novel discovery rate: {novel_rate:.1f}%\n\n")
            
            f.write("CONFIDENCE BREAKDOWN:\n")
            f.write(f"  High confidence: {stats['high_confidence_pairs']}\n")
            f.write(f"  Medium confidence: {stats['medium_confidence_pairs']}\n")
            f.write(f"  Low confidence: {stats['low_confidence_pairs']}\n\n")
            
            f.write("DATABASE COVERAGE:\n")
            for db, count in analysis["database_coverage"].items():
                f.write(f"  {db}: {count} interactions validated\n")
        
        # 2. Validated Interactions Report
        validated_file = output_dir / "validated_interactions.json"
        with open(validated_file, 'w') as f:
            json.dump(analysis["validated_interactions"], f, indent=2)
        
        # 3. Novel Interactions Report  
        novel_file = output_dir / "novel_interactions.json"
        with open(novel_file, 'w') as f:
            json.dump(analysis["novel_interactions"], f, indent=2)
        
        # 4. High Priority Novel Interactions (for experimental validation)
        priority_file = output_dir / "high_priority_novel_interactions.txt"
        with open(priority_file, 'w') as f:
            f.write("High Priority Novel Interactions for Experimental Validation\n")
            f.write("=" * 65 + "\n\n")
            f.write("These interactions have high AI confidence but no database validation.\n")
            f.write("They represent the most promising candidates for novel discoveries.\n\n")
            
            for i, interaction in enumerate(analysis["high_priority_novel"], 1):
                gene_a, gene_b = interaction["gene_pair"]
                max_prob = interaction["max_ai_probability"]
                
                f.write(f"{i}. {gene_a} - {gene_b}\n")
                f.write(f"   AI Confidence: {max_prob:.3f}\n")
                f.write(f"   Predictions: {len(interaction['ai_predictions'])}\n")
                
                # Show top AI prediction
                top_pred = max(interaction["ai_predictions"], 
                             key=lambda x: x["probability"])
                f.write(f"   Top Mechanism: {top_pred['mechanism']}\n")
                f.write(f"   Evidence: {top_pred['evidence'][:100]}...\n")
                f.write(f"   Source: {Path(top_pred['source_file']).name}\n\n")
        
        # 5. Research Recommendations
        recommendations_file = output_dir / "research_recommendations.md"
        with open(recommendations_file, 'w') as f:
            f.write("# Research Recommendations from Cache Validation\n\n")
            
            f.write("## Summary\n")
            stats = analysis["summary_statistics"]
            f.write(f"- **{stats['validated_pairs']} validated interactions** confirm AI accuracy against known biology\n")
            f.write(f"- **{stats['novel_pairs']} novel interactions** represent potential discoveries\n")
            f.write(f"- **{len(analysis['high_priority_novel'])} high-priority novel interactions** merit experimental validation\n\n")
            
            f.write("## Experimental Validation Priorities\n\n")
            f.write("### Immediate Priority (High AI Confidence + Novel)\n")
            for interaction in analysis["high_priority_novel"][:5]:
                gene_a, gene_b = interaction["gene_pair"]
                prob = interaction["max_ai_probability"]
                f.write(f"- **{gene_a} - {gene_b}** (AI confidence: {prob:.3f})\n")
            
            f.write("\n### Literature Review Candidates\n")
            f.write("These validated interactions should be cross-referenced with recent literature:\n")
            for interaction in analysis["validated_interactions"][:5]:
                gene_a, gene_b = interaction["gene_pair"]
                conf = interaction["confidence_level"]
                f.write(f"- **{gene_a} - {gene_b}** ({conf} confidence)\n")
            
            f.write("\n## Database Integration Recommendations\n")
            db_coverage = analysis["database_coverage"]
            if not db_coverage:
                f.write("- Consider obtaining API access for BioGRID and KEGG databases\n")
                f.write("- STRING database is providing good coverage\n")
            else:
                f.write("- Current database coverage:\n")
                for db, count in db_coverage.items():
                    f.write(f"  - {db}: {count} interactions\n")
        
        logger.info(f"Generated comprehensive reports in: {output_dir}")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Validate all interactions in INTERACTION_CACHE")
    parser.add_argument("--cache-dir", default="INTERACTION_CACHE", 
                        help="Path to INTERACTION_CACHE directory")
    parser.add_argument("--output", help="Output directory for validation results")
    parser.add_argument("--summary-only", action="store_true",
                        help="Generate only summary statistics")
    parser.add_argument("--min-confidence", type=float, default=0.4,
                        help="Minimum confidence threshold for validation")
    parser.add_argument("--max-interactions", type=int,
                        help="Maximum number of interactions to validate")
    parser.add_argument("--novel-priority", action="store_true",
                        help="Focus analysis on identifying novel interactions")
    
    args = parser.parse_args()
    
    # Set up output directory
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"cache_validation_{timestamp}"
    
    output_dir = Path(args.output)
    
    logger.info("Starting INTERACTION_CACHE validation analysis...")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = CacheValidationAnalyzer(cache_dir=args.cache_dir)
    
    # Discover interaction files
    interaction_files = analyzer.discover_interaction_files()
    if not interaction_files:
        logger.error("No interaction files found in cache directory")
        sys.exit(1)
    
    # Extract gene pairs
    logger.info("Extracting gene pairs from cache files...")
    gene_pair_data = analyzer.extract_gene_pairs_from_cache(interaction_files)
    
    if not gene_pair_data:
        logger.error("No gene pairs extracted from cache files")
        sys.exit(1)
    
    # Validate gene pairs
    logger.info("Starting database validation...")
    validation_results = analyzer.validate_gene_pairs(
        gene_pair_data, 
        max_interactions=args.max_interactions
    )
    
    # Analyze results
    logger.info("Analyzing validation results...")
    analysis = analyzer.analyze_validation_results(validation_results)
    
    # Generate reports
    if not args.summary_only:
        logger.info("Generating comprehensive reports...")
        report_dir = analyzer.generate_reports(analysis, output_dir)
    
    # Print summary to console
    stats = analysis["summary_statistics"]
    print("\n" + "=" * 80)
    print("INTERACTION_CACHE VALIDATION RESULTS")
    print("=" * 80)
    print(f"Total gene pairs analyzed: {stats['total_gene_pairs']}")
    print(f"Validated (known) interactions: {stats['validated_pairs']}")
    print(f"Novel (unvalidated) interactions: {stats['novel_pairs']}")
    
    if stats['total_gene_pairs'] > 0:
        val_rate = (stats['validated_pairs'] / stats['total_gene_pairs']) * 100
        novel_rate = (stats['novel_pairs'] / stats['total_gene_pairs']) * 100
        print(f"Validation rate: {val_rate:.1f}%")
        print(f"Novel discovery rate: {novel_rate:.1f}%")
    
    print(f"\nHigh-priority novel interactions: {len(analysis['high_priority_novel'])}")
    print(f"Database coverage: {dict(analysis['database_coverage'])}")
    
    if not args.summary_only:
        print(f"\nDetailed reports saved to: {output_dir}")
        print("\nKey files:")
        print(f"  - {output_dir}/validation_summary.txt")
        print(f"  - {output_dir}/high_priority_novel_interactions.txt")
        print(f"  - {output_dir}/research_recommendations.md")

if __name__ == "__main__":
    main()