#!/bin/bash
#
# Random Gene Set Analysis Workflow
# ==================================
# 
# This script runs a complete analysis pipeline for a random set of gene pairs:
# 1. Generate random gene pairs from a CSV/TSV file
# 2. Process all pairs with gene_chain_v1.py
# 3. Compare results across pairs
# 4. Generate summary reports
#
# Usage: ./workflow_random_gene_set.sh INPUT_GENE_FILE NUM_PAIRS [MODEL] [PATHS]
# Example: ./workflow_random_gene_set.sh genes.csv 20 gpt-4.1 3
#

set -e  # Exit on any error

# Function to print colored output
print_step() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_substep() {
    echo ""
    echo ">>> $1"
}

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 INPUT_GENE_FILE NUM_PAIRS [MODEL] [PATHS]"
    echo "Example: $0 genes.csv 20 gpt-4.1 3"
    echo ""
    echo "Required:"
    echo "  INPUT_GENE_FILE    CSV/TSV file containing gene names"
    echo "  NUM_PAIRS          Number of random pairs to generate"
    echo ""
    echo "Optional:"
    echo "  MODEL              AI model to use (default: gpt-4.1)"
    echo "  PATHS              Number of paths to generate per pair (default: 3)"
    echo ""
    echo "The input file should have gene names in the first column or a column named 'Gene'"
    exit 1
fi

# Parse arguments
INPUT_GENES="$1"
NUM_PAIRS="$2"
MODEL="${3:-gpt-4.1}"
PATHS="${4:-3}"

# Check if input file exists
if [ ! -f "$INPUT_GENES" ]; then
    echo "Error: Input gene file '$INPUT_GENES' not found"
    exit 1
fi

# Validate NUM_PAIRS is a number
if ! [[ "$NUM_PAIRS" =~ ^[0-9]+$ ]]; then
    echo "Error: NUM_PAIRS must be a positive integer"
    exit 1
fi

# Check for required environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

print_step "Random Gene Set Analysis Workflow"
echo "Input gene file: $INPUT_GENES"
echo "Number of pairs: $NUM_PAIRS"
echo "Model: $MODEL"
echo "Paths per pair: $PATHS"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="random_gene_analysis_$TIMESTAMP"
mkdir -p "$RUN_DIR"

print_step "Step 1: Generate random gene pairs"
print_substep "Creating $NUM_PAIRS random gene pairs from input file..."

PAIRS_FILE="$RUN_DIR/random_pairs.txt"
echo "Running: python random_gene_pairs.py --input-file $INPUT_GENES --num-pairs $NUM_PAIRS --output-file $PAIRS_FILE"
python random_gene_pairs.py --input-file "$INPUT_GENES" --num-pairs "$NUM_PAIRS" --output-file "$PAIRS_FILE"

if [ $? -eq 0 ]; then
    echo "✓ Step 1 completed successfully"
    echo "  Generated: $PAIRS_FILE"
    
    # Show the generated pairs
    print_substep "Generated gene pairs:"
    cat "$PAIRS_FILE" | head -10
    if [ $(wc -l < "$PAIRS_FILE") -gt 10 ]; then
        echo "... and $(($(wc -l < "$PAIRS_FILE") - 10)) more pairs"
    fi
else
    echo "✗ Step 1 failed"
    exit 1
fi

print_step "Step 2: Process all pairs with interaction analysis"
print_substep "Analyzing interactions for all $NUM_PAIRS gene pairs..."

echo "Running: python gene_chain_v1.py --input-file $PAIRS_FILE --model $MODEL --paths $PATHS"
python gene_chain_v1.py --input-file "$PAIRS_FILE" --model "$MODEL" --paths "$PATHS"

if [ $? -eq 0 ]; then
    echo "✓ Step 2 completed successfully"
    
    # Count generated interaction files
    INTERACTION_COUNT=$(find INTERACTION_CACHE/ -name "*interactions.json" -newer "$PAIRS_FILE" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Generated $INTERACTION_COUNT new interaction analysis files"
else
    echo "✗ Step 2 failed"
    exit 1
fi

print_step "Step 3: Generate pathway reports"
print_substep "Creating comprehensive pathway reports..."

PATHWAY_REPORT="$RUN_DIR/random_set_pathway_report.md"
echo "Running: python gene_pathway_report.py --input-file $PAIRS_FILE --model $MODEL --paths $PATHS --output $PATHWAY_REPORT"
python gene_pathway_report.py --input-file "$PAIRS_FILE" --model "$MODEL" --paths "$PATHS" --output "$PATHWAY_REPORT"

if [ $? -eq 0 ]; then
    echo "✓ Step 3 completed successfully"
    echo "  Generated: $PATHWAY_REPORT"
else
    echo "✗ Step 3 failed"
    exit 1
fi

print_step "Step 4: Extract and compare pathway data"
print_substep "Converting reports to structured data..."

# Extract structured pathway data
JSON_OUTPUT="$RUN_DIR/pathway_data.json"
echo "Running: python extract_pathways.py $PATHWAY_REPORT --output $JSON_OUTPUT --format json"
python extract_pathways.py "$PATHWAY_REPORT" --output "$JSON_OUTPUT" --format json

if [ $? -eq 0 ]; then
    echo "✓ Pathway extraction completed"
    echo "  Generated: $JSON_OUTPUT"
else
    echo "⚠ Pathway extraction failed"
fi

print_substep "Comparing pathways across all pairs..."

# Look for pathway JSON files to compare
PATHWAY_FILES=$(find . INTERACTION_CACHE/ -name "*pathways.json" -newer "$PAIRS_FILE" 2>/dev/null)
PATHWAY_COUNT=$(echo "$PATHWAY_FILES" | wc -l | tr -d ' ')

if [ "$PATHWAY_COUNT" -gt 0 ]; then
    COMPARISON_DIR="$RUN_DIR/pathway_comparisons"
    echo "Running: python compare_pathways.py --output-dir $COMPARISON_DIR"
    python compare_pathways.py --output-dir "$COMPARISON_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ Pathway comparison completed"
        echo "  Generated comparison files in: $COMPARISON_DIR"
    else
        echo "⚠ Pathway comparison failed"
    fi
else
    echo "⚠ No pathway files found for comparison"
fi

print_step "Step 5: Analyze secondary genes and interactions"
print_substep "Identifying key intermediate genes across all pathways..."

SECONDARY_OUTPUT="$RUN_DIR/secondary_genes_analysis.json"
echo "Running: python extract_secondary_genes.py --output $SECONDARY_OUTPUT --min-occurrences 2"
python extract_secondary_genes.py --output "$SECONDARY_OUTPUT" --min-occurrences 2

if [ $? -eq 0 ]; then
    echo "✓ Secondary genes analysis completed"
    echo "  Generated: $SECONDARY_OUTPUT"
    
    # Generate tentative interactions summary
    print_substep "Analyzing tentative interactions (uncertainty patterns)..."
    TENTATIVE_OUTPUT="$RUN_DIR/tentative_interactions_summary.txt"
    echo "Running: python extract_secondary_genes.py --only-tentative --min-occurrences 2"
    python extract_secondary_genes.py --only-tentative --min-occurrences 2 > "$TENTATIVE_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "✓ Tentative interactions analysis completed"
        echo "  Generated: $TENTATIVE_OUTPUT"
    fi
else
    echo "⚠ Secondary genes analysis failed"
fi

print_step "Step 6: Generate summary statistics"
print_substep "Creating analysis summary and statistics..."

SUMMARY_FILE="$RUN_DIR/analysis_summary.txt"

{
    echo "Random Gene Set Analysis Summary"
    echo "================================"
    echo "Generated on: $(date)"
    echo "Input file: $INPUT_GENES"
    echo "Number of random pairs: $NUM_PAIRS"
    echo "Model used: $MODEL"
    echo "Paths per pair: $PATHS"
    echo ""
    
    echo "Files generated:"
    echo "  - Gene pairs: $PAIRS_FILE"
    echo "  - Pathway report: $PATHWAY_REPORT"
    echo "  - Structured data: $JSON_OUTPUT"
    echo "  - Secondary genes: $SECONDARY_OUTPUT"
    echo ""
    
    echo "Interaction files created:"
    TOTAL_INTERACTIONS=$(find INTERACTION_CACHE/ -name "*interactions.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total interaction files: $TOTAL_INTERACTIONS"
    
    echo ""
    echo "Gene pair statistics:"
    echo "  Requested pairs: $NUM_PAIRS"
    echo "  Generated pairs: $(wc -l < "$PAIRS_FILE" 2>/dev/null || echo "0")"
    
    if [ -f "$SECONDARY_OUTPUT" ]; then
        echo ""
        echo "Secondary gene statistics:"
        # Extract some basic stats from the JSON if possible
        if command -v jq >/dev/null 2>&1; then
            SECONDARY_COUNT=$(jq '.secondary_genes | length' "$SECONDARY_OUTPUT" 2>/dev/null || echo "unknown")
            echo "  Unique secondary genes found: $SECONDARY_COUNT"
        else
            echo "  Secondary genes analysis available in: $SECONDARY_OUTPUT"
        fi
    fi
    
    echo ""
    echo "Files location: $RUN_DIR/"
    
} > "$SUMMARY_FILE"

echo "✓ Summary generated: $SUMMARY_FILE"

print_step "Analysis Complete!"
echo "All files are organized in: $RUN_DIR/"
echo ""

# Display the summary
cat "$SUMMARY_FILE"

echo ""
echo "Generated files:"
echo ""

echo "Input and configuration:"
if [ -f "$PAIRS_FILE" ]; then
    echo "  ✓ $PAIRS_FILE (random gene pairs)"
fi

echo ""
echo "Analysis reports:"
if [ -f "$PATHWAY_REPORT" ]; then
    echo "  ✓ $PATHWAY_REPORT (comprehensive pathway analysis)"
fi
if [ -f "$SUMMARY_FILE" ]; then
    echo "  ✓ $SUMMARY_FILE (analysis summary)"
fi

echo ""
echo "Structured data:"
if [ -f "$JSON_OUTPUT" ]; then
    echo "  ✓ $JSON_OUTPUT (structured pathway data)"
fi
if [ -f "$SECONDARY_OUTPUT" ]; then
    echo "  ✓ $SECONDARY_OUTPUT (secondary genes analysis)"
fi
if [ -f "$RUN_DIR/tentative_interactions_summary.txt" ]; then
    echo "  ✓ $RUN_DIR/tentative_interactions_summary.txt"
fi

echo ""
echo "Comparative analysis:"
if [ -d "$RUN_DIR/pathway_comparisons" ]; then
    echo "  ✓ $RUN_DIR/pathway_comparisons/ (pathway comparison results)"
fi

echo ""
echo "Raw data:"
echo "  ✓ INTERACTION_CACHE/ (all interaction data files)"

echo ""
echo "Random gene set analysis workflow completed successfully!"
echo ""
echo "Next steps:"
echo "  - Review the pathway report: $PATHWAY_REPORT"
echo "  - Check secondary genes for key network hubs: $SECONDARY_OUTPUT"
echo "  - Examine pathway comparisons for common patterns"
echo "  - Look for tentative interactions needing validation"