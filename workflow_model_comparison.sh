#!/bin/bash
#
# Model Comparison Analysis Workflow
# ===================================
# 
# This script runs a comprehensive model comparison study:
# 1. Compare predictions from multiple AI models on the same gene pairs
# 2. Extract and format comparative analysis results
# 3. Generate detailed comparison reports
#
# Usage: ./workflow_model_comparison.sh NUM_PAIRS [MODEL1,MODEL2,...] [PATHS]
# Example: ./workflow_model_comparison.sh 10 "gpt41,claude,llama" 3
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
    echo "Usage: $0 NUM_PAIRS [MODEL_LIST] [PATHS]"
    echo "Example: $0 10 \"gpt41,claude,llama\" 3"
    echo ""
    echo "Required:"
    echo "  NUM_PAIRS          Number of gene pairs to compare across models"
    echo ""
    echo "Optional:"
    echo "  MODEL_LIST         Comma-separated list of model shortnames (default: \"gpt41,claude\")"
    echo "  PATHS              Number of paths to generate per pair (default: 3)"
    echo ""
    echo "Available model shortnames are defined in model_servers.yaml"
    echo "Use 'python model_config.py --list' to see available models"
    exit 1
fi

# Parse arguments
NUM_PAIRS="$1"
MODEL_LIST="${2:-gpt41,claude}"
PATHS="${3:-3}"

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

print_step "Model Comparison Analysis Workflow"
echo "Number of pairs: $NUM_PAIRS"
echo "Models to compare: $MODEL_LIST"
echo "Paths per pair: $PATHS"
echo ""

# Convert comma-separated model list to space-separated for processing
MODELS_ARRAY=($(echo "$MODEL_LIST" | tr ',' ' '))
echo "Models selected: ${MODELS_ARRAY[*]}"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="model_comparison_$TIMESTAMP"
mkdir -p "$RUN_DIR"

print_step "Step 1: Check available models and endpoints"
print_substep "Verifying model configurations and endpoint availability..."

echo "Available models in configuration:"
if python model_config.py --list 2>/dev/null; then
    echo "✓ Model configuration loaded successfully"
else
    echo "⚠ Could not load model configuration. Using default models."
fi

echo ""
echo "Testing endpoint availability for selected models..."
for model in "${MODELS_ARRAY[@]}"; do
    echo "  Testing model: $model"
done

print_step "Step 2: Run model comparison analysis"
print_substep "Comparing predictions across multiple models for $NUM_PAIRS random gene pairs..."

COMPARISON_OUTPUT="$RUN_DIR/model_comparison_results.json"
COMPARISON_ARGS="--pairs $NUM_PAIRS --paths $PATHS --out $COMPARISON_OUTPUT"

# Add model list if specified
if [ "$MODEL_LIST" != "gpt41,claude" ]; then
    # Convert back to individual model arguments
    for model in "${MODELS_ARRAY[@]}"; do
        COMPARISON_ARGS="$COMPARISON_ARGS --models $model"
    done
else
    # Use default models
    COMPARISON_ARGS="$COMPARISON_ARGS --models gpt41 --models claude"
fi

echo "Running: python compare_model_predictions.py $COMPARISON_ARGS"
python compare_model_predictions.py $COMPARISON_ARGS

if [ $? -eq 0 ]; then
    echo "✓ Step 2 completed successfully"
    echo "  Generated: $COMPARISON_OUTPUT"
    
    # Check if partial results exist (for interrupted runs)
    if [ -f "$COMPARISON_OUTPUT.partial" ]; then
        echo "  Partial results: $COMPARISON_OUTPUT.partial"
    fi
else
    echo "✗ Step 2 failed"
    echo "Note: Check if all models are properly configured and accessible"
    exit 1
fi

print_step "Step 3: Extract and format comparative analyses"
print_substep "Processing GPT-4.1 comparative analysis results..."

if [ -f "$COMPARISON_OUTPUT" ]; then
    # Extract comparative analyses in multiple formats
    print_substep "Generating text format analysis..."
    TEXT_ANALYSIS="$RUN_DIR/comparative_analyses.txt"
    echo "Running: python extract_comparative_analysis.py --input $COMPARISON_OUTPUT --output-dir $RUN_DIR --format text --single-file --output-file comparative_analyses.txt"
    python extract_comparative_analysis.py --input "$COMPARISON_OUTPUT" --output-dir "$RUN_DIR" --format text --single-file --output-file "comparative_analyses.txt"
    
    if [ $? -eq 0 ]; then
        echo "✓ Text analysis extraction completed"
        echo "  Generated: $TEXT_ANALYSIS"
    fi
    
    print_substep "Generating markdown format analysis..."
    MD_ANALYSIS="$RUN_DIR/comparative_analyses.md"
    echo "Running: python extract_comparative_analysis.py --input $COMPARISON_OUTPUT --output-dir $RUN_DIR --format md --single-file --output-file comparative_analyses.md"
    python extract_comparative_analysis.py --input "$COMPARISON_OUTPUT" --output-dir "$RUN_DIR" --format md --single-file --output-file "comparative_analyses.md"
    
    if [ $? -eq 0 ]; then
        echo "✓ Markdown analysis extraction completed"
        echo "  Generated: $MD_ANALYSIS"
    fi
    
    print_substep "Generating HTML format analysis..."
    HTML_ANALYSIS="$RUN_DIR/comparative_analyses.html"
    echo "Running: python extract_comparative_analysis.py --input $COMPARISON_OUTPUT --output-dir $RUN_DIR --format html --single-file --output-file comparative_analyses.html"
    python extract_comparative_analysis.py --input "$COMPARISON_OUTPUT" --output-dir "$RUN_DIR" --format html --single-file --output-file "comparative_analyses.html"
    
    if [ $? -eq 0 ]; then
        echo "✓ HTML analysis extraction completed"
        echo "  Generated: $HTML_ANALYSIS"
    fi
else
    echo "⚠ Comparison results file not found, skipping analysis extraction"
fi

print_step "Step 4: Generate summary statistics and insights"
print_substep "Creating comprehensive comparison summary..."

SUMMARY_FILE="$RUN_DIR/comparison_summary.txt"

{
    echo "Model Comparison Analysis Summary"
    echo "================================="
    echo "Generated on: $(date)"
    echo "Number of gene pairs analyzed: $NUM_PAIRS"
    echo "Models compared: $MODEL_LIST"
    echo "Paths per pair: $PATHS"
    echo ""
    
    echo "Analysis files generated:"
    echo "  - Raw comparison data: $COMPARISON_OUTPUT"
    if [ -f "$TEXT_ANALYSIS" ]; then
        echo "  - Text analysis: $TEXT_ANALYSIS"
    fi
    if [ -f "$MD_ANALYSIS" ]; then
        echo "  - Markdown analysis: $MD_ANALYSIS"
    fi
    if [ -f "$HTML_ANALYSIS" ]; then
        echo "  - HTML analysis: $HTML_ANALYSIS"
    fi
    echo ""
    
    if [ -f "$COMPARISON_OUTPUT" ]; then
        echo "Comparison statistics:"
        # Try to extract basic statistics from the JSON
        if command -v jq >/dev/null 2>&1; then
            echo "  Models analyzed: $(jq -r '.models[]' "$COMPARISON_OUTPUT" 2>/dev/null | paste -sd "," - || echo "Could not extract")"
            echo "  Total files analyzed: $(jq -r '.summary.total_files_analyzed' "$COMPARISON_OUTPUT" 2>/dev/null || echo "Could not extract")"
            echo "  Files with errors: $(jq -r '.summary.errors' "$COMPARISON_OUTPUT" 2>/dev/null || echo "Could not extract")"
            echo "  Full agreement count: $(jq -r '.summary.full_agreement_count' "$COMPARISON_OUTPUT" 2>/dev/null || echo "Could not extract")"
            echo "  Partial agreement count: $(jq -r '.summary.partial_agreement_count' "$COMPARISON_OUTPUT" 2>/dev/null || echo "Could not extract")"
            echo "  Disagreement count: $(jq -r '.summary.disagreement_count' "$COMPARISON_OUTPUT" 2>/dev/null || echo "Could not extract")"
        else
            echo "  Full statistics available in: $COMPARISON_OUTPUT"
            echo "  (Install jq for detailed summary here)"
        fi
        echo ""
    fi
    
    echo "Interaction files analyzed:"
    INTERACTION_COUNT=$(find INTERACTION_CACHE/ -name "*interactions.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total interaction files in cache: $INTERACTION_COUNT"
    
    echo ""
    echo "All files are located in: $RUN_DIR/"
    
} > "$SUMMARY_FILE"

echo "✓ Summary generated: $SUMMARY_FILE"

print_step "Step 5: Generate insights and recommendations"
print_substep "Analyzing model agreement patterns..."

INSIGHTS_FILE="$RUN_DIR/insights_and_recommendations.txt"

{
    echo "Model Comparison Insights and Recommendations"
    echo "============================================="
    echo "Generated on: $(date)"
    echo ""
    
    echo "Key Questions to Explore:"
    echo "  1. Which models show the highest agreement on pathway predictions?"
    echo "  2. Are there systematic differences in the types of interactions predicted?"
    echo "  3. Which gene pairs show the most model disagreement (need further validation)?"
    echo "  4. Do models differ in their confidence/probability assignments?"
    echo "  5. Are there patterns in evidence quality across different models?"
    echo ""
    
    echo "Files to Review for Insights:"
    if [ -f "$MD_ANALYSIS" ]; then
        echo "  - Start with: $MD_ANALYSIS (most readable format)"
    fi
    if [ -f "$COMPARISON_OUTPUT" ]; then
        echo "  - Detailed data: $COMPARISON_OUTPUT (machine-readable)"
    fi
    if [ -f "$HTML_ANALYSIS" ]; then
        echo "  - Web view: $HTML_ANALYSIS (open in browser)"
    fi
    echo ""
    
    echo "Recommended Next Steps:"
    echo "  1. Review comparative analyses for each gene pair"
    echo "  2. Identify gene pairs with high model disagreement"
    echo "  3. Cross-reference predictions with literature"
    echo "  4. Consider experimental validation for disputed interactions"
    echo "  5. Document model-specific biases or strengths observed"
    echo ""
    
    echo "Model Performance Evaluation:"
    echo "  - Look for models that consistently provide well-supported evidence"
    echo "  - Note models that tend to be more conservative vs. speculative"
    echo "  - Identify models that excel at different types of interactions"
    echo "  - Consider computational cost vs. prediction quality trade-offs"
    
} > "$INSIGHTS_FILE"

echo "✓ Insights generated: $INSIGHTS_FILE"

print_step "Analysis Complete!"
echo "All files are organized in: $RUN_DIR/"
echo ""

# Display the summary
cat "$SUMMARY_FILE"

echo ""
echo "Generated files:"
echo ""

echo "Core results:"
if [ -f "$COMPARISON_OUTPUT" ]; then
    echo "  ✓ $COMPARISON_OUTPUT (raw comparison data)"
fi
if [ -f "$SUMMARY_FILE" ]; then
    echo "  ✓ $SUMMARY_FILE (analysis summary)"
fi

echo ""
echo "Formatted analyses:"
if [ -f "$TEXT_ANALYSIS" ]; then
    echo "  ✓ $TEXT_ANALYSIS (text format)"
fi
if [ -f "$MD_ANALYSIS" ]; then
    echo "  ✓ $MD_ANALYSIS (markdown format)"
fi
if [ -f "$HTML_ANALYSIS" ]; then
    echo "  ✓ $HTML_ANALYSIS (HTML format)"
fi

echo ""
echo "Guidance:"
if [ -f "$INSIGHTS_FILE" ]; then
    echo "  ✓ $INSIGHTS_FILE (insights and recommendations)"
fi

echo ""
echo "Supporting data:"
echo "  ✓ INTERACTION_CACHE/ (all interaction data files)"

echo ""
echo "Model comparison workflow completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review the formatted analyses to understand model differences"
echo "  2. Identify gene pairs requiring further validation"
echo "  3. Document patterns in model behavior for future reference"
echo "  4. Consider running additional comparisons with different parameters"

if [ -f "$MD_ANALYSIS" ]; then
    echo ""
    echo "Quick start: Review the markdown analysis file:"
    echo "  $MD_ANALYSIS"
fi