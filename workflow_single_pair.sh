#!/bin/bash
#
# Single Gene Pair Analysis Workflow
# ===================================
# 
# This script runs a complete analysis pipeline for a single gene pair:
# 1. Generate basic interaction data
# 2. Create detailed analysis report  
# 3. Generate biological summary
#
# Usage: ./workflow_single_pair.sh GENE1 GENE2 [MODEL] [PATHS]
# Example: ./workflow_single_pair.sh TP53 EGFR gpt-4.1 3
#

set -e  # Exit on any error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 GENE1 GENE2 [MODEL] [PATHS]"
    echo "Example: $0 TP53 EGFR gpt-4.1 3"
    echo ""
    echo "Required:"
    echo "  GENE1     First gene/protein name"
    echo "  GENE2     Second gene/protein name"
    echo ""
    echo "Optional:"
    echo "  MODEL     AI model to use (default: gpt-4.1)"
    echo "  PATHS     Number of paths to generate (default: 3)"
    exit 1
fi

# Parse arguments
GENE1="$1"
GENE2="$2"
MODEL="${3:-gpt-4.1}"
PATHS="${4:-3}"

# Check for required environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "=========================================="
echo "Single Gene Pair Analysis Workflow"
echo "=========================================="
echo "Gene 1: $GENE1"
echo "Gene 2: $GENE2"
echo "Model: $MODEL"
echo "Paths: $PATHS"
echo ""

# Step 1: Generate basic interaction data
echo "Step 1: Generating interaction data..."
echo "Running: python gene_chain_v1.py $GENE1 $GENE2 --model $MODEL --paths $PATHS"
python gene_chain_v1.py "$GENE1" "$GENE2" --model "$MODEL" --paths "$PATHS"

if [ $? -eq 0 ]; then
    echo "✓ Step 1 completed successfully"
else
    echo "✗ Step 1 failed"
    exit 1
fi
echo ""

# Step 2: Generate detailed analysis report
echo "Step 2: Generating detailed pathway report..."
echo "Running: python gene_pathway_report.py $GENE1 $GENE2 --model $MODEL --paths $PATHS"
python gene_pathway_report.py "$GENE1" "$GENE2" --model "$MODEL" --paths "$PATHS"

if [ $? -eq 0 ]; then
    echo "✓ Step 2 completed successfully"
else
    echo "✗ Step 2 failed"
    exit 1
fi
echo ""

# Step 3: Generate biological summary
echo "Step 3: Generating biological summary..."
INTERACTION_FILE="network_${GENE1}_${GENE2}_interactions.json"

# Check if interaction file exists (in current directory or cache)
if [ -f "$INTERACTION_FILE" ]; then
    INPUT_FILE="$INTERACTION_FILE"
elif [ -f "INTERACTION_CACHE/$INTERACTION_FILE" ]; then
    INPUT_FILE="INTERACTION_CACHE/$INTERACTION_FILE"
else
    echo "Warning: Could not find interaction file, trying alternative names..."
    # Try to find any interaction file for these genes
    INPUT_FILE=$(find . INTERACTION_CACHE/ -name "*${GENE1}*${GENE2}*interactions.json" -o -name "*${GENE2}*${GENE1}*interactions.json" 2>/dev/null | head -1)
    
    if [ -z "$INPUT_FILE" ]; then
        echo "Warning: No interaction file found, skipping biological summary"
        INPUT_FILE=""
    fi
fi

if [ -n "$INPUT_FILE" ]; then
    echo "Running: python interactions_to_summary.py --input-file $INPUT_FILE"
    python interactions_to_summary.py --input-file "$INPUT_FILE" --model "$MODEL"
    
    if [ $? -eq 0 ]; then
        echo "✓ Step 3 completed successfully"
    else
        echo "✗ Step 3 failed"
        exit 1
    fi
else
    echo "⚠ Step 3 skipped (no interaction file found)"
fi
echo ""

# Summary of outputs
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Generated files:"

# List the generated files
echo ""
echo "Interaction data:"
for file in network_${GENE1}_${GENE2}_interactions.json ${GENE1}_${GENE2}_interactions.json INTERACTION_CACHE/network_${GENE1}_${GENE2}_interactions.json; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "Visualization files:"
for file in network_${GENE1}_${GENE2}.dot ${GENE1}_${GENE2}.dot network_${GENE1}_${GENE2}.png ${GENE1}_${GENE2}.png INTERACTION_CACHE/network_${GENE1}_${GENE2}.dot INTERACTION_CACHE/network_${GENE1}_${GENE2}.png; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "Analysis reports:"
for file in gene_pathway_report_${GENE1}_${GENE2}.md; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "Biological summaries:"
for file in summary_${GENE1}_${GENE2}.txt INTERACTION_CACHE/summary_${GENE1}_${GENE2}.txt; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    fi
done

echo ""
echo "Analysis workflow completed successfully!"
echo "Check the generated files above for results."