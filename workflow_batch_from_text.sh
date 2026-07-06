#!/bin/bash
#
# Batch Analysis from Text Workflow
# ==================================
# 
# This script runs a complete batch analysis pipeline starting from biological text:
# 1. Extract entities and generate gene pairs from text
# 2. Generate reports for all pairs
# 3. Extract structured data from reports
# 4. Analyze intermediate genes in pathways
#
# Usage: ./workflow_batch_from_text.sh INPUT_TEXT_FILE [MODEL] [PATHS]
# Example: ./workflow_batch_from_text.sh biological_paper.txt gpt-4.1 3
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
if [ $# -lt 1 ]; then
    echo "Usage: $0 INPUT_TEXT_FILE [MODEL] [PATHS]"
    echo "Example: $0 biological_paper.txt gpt-4.1 3"
    echo ""
    echo "Required:"
    echo "  INPUT_TEXT_FILE    Text file containing biological context"
    echo ""
    echo "Optional:"
    echo "  MODEL              AI model to use (default: gpt-4.1)"
    echo "  PATHS              Number of paths to generate per pair (default: 3)"
    exit 1
fi

# Parse arguments
INPUT_TEXT="$1"
MODEL="${2:-gpt-4.1}"
PATHS="${3:-3}"

# Check if input file exists
if [ ! -f "$INPUT_TEXT" ]; then
    echo "Error: Input file '$INPUT_TEXT' not found"
    exit 1
fi

# Check for required environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

print_step "Batch Analysis from Text Workflow"
echo "Input text file: $INPUT_TEXT"
echo "Model: $MODEL"
echo "Paths per pair: $PATHS"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="batch_analysis_$TIMESTAMP"
mkdir -p "$RUN_DIR"

print_step "Step 1: Extract entities and generate gene pairs"
print_substep "Extracting gene/protein entities from text and generating pairs..."

echo "Running: python context_to_pairs.py -i $INPUT_TEXT --model $MODEL --model-for-chain $MODEL --paths $PATHS"
python context_to_pairs.py -i "$INPUT_TEXT" --model "$MODEL" --model-for-chain "$MODEL" --paths "$PATHS"

if [ $? -eq 0 ]; then
    echo "✓ Step 1 completed successfully"
    # Move generated pairs file to run directory
    if [ -f "pairs.txt" ]; then
        cp "pairs.txt" "$RUN_DIR/"
        echo "  Generated pairs file: $RUN_DIR/pairs.txt"
    fi
else
    echo "✗ Step 1 failed"
    exit 1
fi

print_step "Step 2: Generate pathway reports for all pairs"
print_substep "Creating detailed analysis reports for all gene pairs..."

if [ ! -f "pairs.txt" ]; then
    echo "Error: pairs.txt not found. Step 1 may have failed."
    exit 1
fi

# Count the number of pairs
PAIR_COUNT=$(wc -l < pairs.txt | tr -d ' ')
echo "Found $PAIR_COUNT gene pairs to analyze"

OUTPUT_REPORT="$RUN_DIR/multiple_pathway_report.md"
echo "Running: python gene_pathway_report.py --input-file pairs.txt --model $MODEL --paths $PATHS --output $OUTPUT_REPORT"
python gene_pathway_report.py --input-file "pairs.txt" --model "$MODEL" --paths "$PATHS" --output "$OUTPUT_REPORT"

if [ $? -eq 0 ]; then
    echo "✓ Step 2 completed successfully"
    echo "  Generated report: $OUTPUT_REPORT"
else
    echo "✗ Step 2 failed"
    exit 1
fi

print_step "Step 3: Extract structured data from reports"
print_substep "Converting markdown reports to structured JSON data..."

if [ -f "$OUTPUT_REPORT" ]; then
    JSON_OUTPUT="$RUN_DIR/pathways_data.json"
    CSV_OUTPUT="$RUN_DIR/pathways_data.csv"
    
    echo "Running: python extract_pathways.py $OUTPUT_REPORT --output $JSON_OUTPUT --format json"
    python extract_pathways.py "$OUTPUT_REPORT" --output "$JSON_OUTPUT" --format json
    
    if [ $? -eq 0 ]; then
        echo "✓ JSON extraction completed"
        echo "  Generated: $JSON_OUTPUT"
        
        # Also generate CSV format
        echo "Running: python extract_pathways.py $OUTPUT_REPORT --output $CSV_OUTPUT --format csv"
        python extract_pathways.py "$OUTPUT_REPORT" --output "$CSV_OUTPUT" --format csv
        
        if [ $? -eq 0 ]; then
            echo "✓ CSV extraction completed"
            echo "  Generated: $CSV_OUTPUT"
        else
            echo "⚠ CSV extraction failed (JSON still available)"
        fi
    else
        echo "✗ Step 3 failed"
        exit 1
    fi
else
    echo "Warning: Report file not found, skipping structured data extraction"
fi

print_step "Step 4: Analyze secondary genes in pathways"
print_substep "Identifying and analyzing intermediate genes..."

# Look for JSON pathway files
PATHWAY_FILES=$(find . INTERACTION_CACHE/ -name "*pathways.json" 2>/dev/null | head -5)

if [ -n "$PATHWAY_FILES" ]; then
    SECONDARY_OUTPUT="$RUN_DIR/secondary_genes_analysis.json"
    echo "Running: python extract_secondary_genes.py --output $SECONDARY_OUTPUT"
    python extract_secondary_genes.py --output "$SECONDARY_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "✓ Step 4 completed successfully"
        echo "  Generated: $SECONDARY_OUTPUT"
        
        # Also run with tentative interactions analysis
        print_substep "Analyzing tentative interactions..."
        echo "Running: python extract_secondary_genes.py --show-tentative --min-occurrences 2"
        python extract_secondary_genes.py --show-tentative --min-occurrences 2 | tee "$RUN_DIR/tentative_interactions_analysis.txt"
        
        if [ $? -eq 0 ]; then
            echo "✓ Tentative interactions analysis completed"
            echo "  Generated: $RUN_DIR/tentative_interactions_analysis.txt"
        fi
    else
        echo "⚠ Step 4 failed (secondary genes analysis)"
    fi
else
    echo "⚠ Step 4 skipped (no pathway JSON files found)"
fi

print_step "Analysis Complete!"
echo "All generated files are in: $RUN_DIR/"
echo ""

# Summary of outputs
echo "Generated files:"
echo ""

echo "Configuration and inputs:"
if [ -f "$RUN_DIR/pairs.txt" ]; then
    echo "  ✓ $RUN_DIR/pairs.txt (generated gene pairs)"
fi

echo ""
echo "Analysis reports:"
if [ -f "$OUTPUT_REPORT" ]; then
    echo "  ✓ $OUTPUT_REPORT (comprehensive pathway report)"
fi

echo ""
echo "Structured data:"
if [ -f "$RUN_DIR/pathways_data.json" ]; then
    echo "  ✓ $RUN_DIR/pathways_data.json (structured pathway data)"
fi
if [ -f "$RUN_DIR/pathways_data.csv" ]; then
    echo "  ✓ $RUN_DIR/pathways_data.csv (tabular pathway data)"
fi

echo ""
echo "Secondary analysis:"
if [ -f "$RUN_DIR/secondary_genes_analysis.json" ]; then
    echo "  ✓ $RUN_DIR/secondary_genes_analysis.json (intermediate genes analysis)"
fi
if [ -f "$RUN_DIR/tentative_interactions_analysis.txt" ]; then
    echo "  ✓ $RUN_DIR/tentative_interactions_analysis.txt (tentative interactions)"
fi

echo ""
echo "Raw interaction data:"
INTERACTION_COUNT=$(find INTERACTION_CACHE/ -name "*interactions.json" 2>/dev/null | wc -l | tr -d ' ')
echo "  ✓ $INTERACTION_COUNT interaction files in INTERACTION_CACHE/"

echo ""
echo "Batch analysis workflow completed successfully!"
echo "Results are organized in: $RUN_DIR/"
echo ""
echo "Next steps:"
echo "  - Review the comprehensive report: $OUTPUT_REPORT"
echo "  - Examine structured data: $RUN_DIR/pathways_data.json"
echo "  - Check secondary genes analysis for key intermediates"
echo "  - Look for tentative interactions that may need further validation"