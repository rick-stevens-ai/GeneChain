#!/bin/bash
#
# Gene Interaction Validation Workflow
# =====================================
# 
# This script validates AI-generated gene interactions against known biological databases:
# 1. Validates interactions against STRING, BioGRID, KEGG, and PubMed
# 2. Generates comprehensive validation reports
# 3. Identifies novel vs. known interactions
# 4. Provides validation metrics and confidence scoring
#
# Usage: ./workflow_validation.sh INPUT [OUTPUT_DIR]
# Examples:
#   ./workflow_validation.sh network_TP53_EGFR_interactions.json
#   ./workflow_validation.sh "*_interactions.json" validation_results
#   ./workflow_validation.sh INTERACTION_CACHE/
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

print_info() { echo -e "\033[0;34m[INFO]\033[0m $1"; }
print_success() { echo -e "\033[0;32m[SUCCESS]\033[0m $1"; }
print_warning() { echo -e "\033[0;33m[WARNING]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 INPUT [OUTPUT_DIR]"
    echo ""
    echo "INPUT can be:"
    echo "  - Single interaction file: network_TP53_EGFR_interactions.json"
    echo "  - File pattern: \"*_interactions.json\""
    echo "  - Directory: INTERACTION_CACHE/"
    echo ""
    echo "OUTPUT_DIR: Directory for validation results (default: validation_results_TIMESTAMP)"
    echo ""
    echo "Examples:"
    echo "  $0 network_TP53_EGFR_interactions.json"
    echo "  $0 \"*_interactions.json\" my_validation"
    echo "  $0 INTERACTION_CACHE/"
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="${2:-validation_results_$(date +%Y%m%d_%H%M%S)}"

print_step "Gene Interaction Validation Workflow"
echo "Input: $INPUT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if input is a directory, file, or pattern
if [ -d "$INPUT" ]; then
    print_info "Input is a directory, processing all interaction files in: $INPUT"
    INPUT_PATTERN="$INPUT/*_interactions.json"
    BATCH_MODE=true
elif [[ "$INPUT" == *"*"* ]]; then
    print_info "Input is a pattern, processing files matching: $INPUT"
    INPUT_PATTERN="$INPUT"
    BATCH_MODE=true
elif [ -f "$INPUT" ]; then
    print_info "Input is a single file: $INPUT"
    INPUT_PATTERN="$INPUT"
    BATCH_MODE=false
else
    print_error "Input not found or invalid: $INPUT"
    exit 1
fi

# Check Python dependencies
print_step "Step 1: Check Dependencies"
print_substep "Checking Python packages..."

MISSING_DEPS=()
python3 -c "import requests" 2>/dev/null || MISSING_DEPS+=("requests")
python3 -c "import pandas" 2>/dev/null || MISSING_DEPS+=("pandas")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_error "Missing required Python packages: ${MISSING_DEPS[*]}"
    echo "Install with: pip install ${MISSING_DEPS[*]}"
    exit 1
fi

# Check optional dependencies
OPTIONAL_DEPS=()
python3 -c "import scipy" 2>/dev/null || OPTIONAL_DEPS+=("scipy")
python3 -c "import bioservices" 2>/dev/null || OPTIONAL_DEPS+=("bioservices")

if [ ${#OPTIONAL_DEPS[@]} -gt 0 ]; then
    print_warning "Optional packages not found: ${OPTIONAL_DEPS[*]}"
    echo "For enhanced functionality, install with: pip install ${OPTIONAL_DEPS[*]}"
fi

print_success "Dependencies checked"

# Count input files
print_step "Step 2: Analyze Input Files"
if [ "$BATCH_MODE" = true ]; then
    FILE_COUNT=$(find . -name "$(basename "$INPUT_PATTERN")" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$FILE_COUNT" -eq 0 ]; then
        print_error "No files found matching pattern: $INPUT_PATTERN"
        exit 1
    fi
    print_info "Found $FILE_COUNT interaction files to validate"
else
    FILE_COUNT=1
    print_info "Processing single file: $INPUT"
fi

# Run validation
print_step "Step 3: Validate Interactions Against Databases"
print_substep "Running comprehensive validation..."

CACHE_DIR="$OUTPUT_DIR/database_cache"
VALIDATION_OUTPUT="$OUTPUT_DIR/validation_results.json"

echo "This will validate interactions against:"
echo "  - STRING database (protein-protein interactions)"
echo "  - BioGRID database (curated interactions)"
echo "  - KEGG database (pathway information)"
echo "  - PubMed literature (publication evidence)"
echo ""

if [ "$BATCH_MODE" = true ]; then
    print_info "Running batch validation..."
    echo "python3 validate_interactions.py --input \"$INPUT_PATTERN\" --batch --output \"$VALIDATION_OUTPUT\" --cache-dir \"$CACHE_DIR\""
    python3 validate_interactions.py --input "$INPUT_PATTERN" --batch --output "$VALIDATION_OUTPUT" --cache-dir "$CACHE_DIR"
else
    print_info "Running single file validation..."
    echo "python3 validate_interactions.py --input \"$INPUT\" --output \"$VALIDATION_OUTPUT\" --cache-dir \"$CACHE_DIR\""
    python3 validate_interactions.py --input "$INPUT" --output "$VALIDATION_OUTPUT" --cache-dir "$CACHE_DIR"
fi

if [ $? -eq 0 ]; then
    print_success "Validation completed successfully"
else
    print_error "Validation failed"
    exit 1
fi

# Generate summary report
print_step "Step 4: Generate Validation Report"
print_substep "Creating human-readable validation report..."

SUMMARY_REPORT="$OUTPUT_DIR/validation_summary.txt"
DETAILED_REPORT="$OUTPUT_DIR/validation_detailed.md"

# Extract summary from JSON and create readable report
python3 -c "
import json
import sys
from datetime import datetime

try:
    with open('$VALIDATION_OUTPUT', 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary_statistics', {})
    metadata = data.get('metadata', {})
    
    # Generate summary report
    with open('$SUMMARY_REPORT', 'w') as f:
        f.write('Gene Interaction Validation Summary Report\\n')
        f.write('==========================================\\n\\n')
        f.write(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\\n')
        f.write(f'Input files: {metadata.get(\"total_files_processed\", \"Unknown\")}\\n')
        f.write(f'Species: {metadata.get(\"species\", \"Unknown\")} (NCBI Taxonomy ID)\\n\\n')
        
        f.write('VALIDATION STATISTICS:\\n')
        f.write(f'  Total interactions analyzed: {summary.get(\"total_interactions\", 0)}\\n')
        f.write(f'  Validated interactions: {summary.get(\"validated_interactions\", 0)}\\n')
        f.write(f'  Novel interactions: {summary.get(\"novel_interactions\", 0)}\\n\\n')
        
        if summary.get('total_interactions', 0) > 0:
            total = summary['total_interactions']
            validated = summary.get('validated_interactions', 0)
            novel = summary.get('novel_interactions', 0)
            val_rate = (validated / total) * 100
            novel_rate = (novel / total) * 100
            f.write(f'  Validation rate: {val_rate:.1f}%\\n')
            f.write(f'  Novel discovery rate: {novel_rate:.1f}%\\n\\n')
        
        f.write('DATABASE COVERAGE:\\n')
        db_coverage = summary.get('database_coverage', {})
        for db, count in db_coverage.items():
            f.write(f'  {db}: {count} interactions\\n')
        
        f.write('\\nCONFIDENCE DISTRIBUTION:\\n')
        conf_dist = summary.get('confidence_distribution', {})
        for conf, count in conf_dist.items():
            f.write(f'  {conf}: {count} interactions\\n')
        
        f.write('\\nFILES ANALYZED:\\n')
        for file_path in metadata.get('input_files', []):
            f.write(f'  - {file_path}\\n')
    
    print('Summary report generated successfully')
    
except Exception as e:
    print(f'Error generating summary report: {e}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Summary report generated: $SUMMARY_REPORT"
else
    print_warning "Could not generate summary report"
fi

# Generate detailed markdown report for novel interactions
python3 -c "
import json
import sys

try:
    with open('$VALIDATION_OUTPUT', 'r') as f:
        data = json.load(f)
    
    results = data.get('validation_results', [])
    
    with open('$DETAILED_REPORT', 'w') as f:
        f.write('# Detailed Validation Results\\n\\n')
        
        # Novel interactions section
        f.write('## Novel Interactions (Potential Discoveries)\\n\\n')
        novel_count = 0
        for result in results:
            if result.get('novel_interaction', False):
                novel_count += 1
                gene_a = result['gene_pair'][0]
                gene_b = result['gene_pair'][1]
                ai_pred = result.get('ai_prediction', {})
                
                f.write(f'### {gene_a} - {gene_b}\\n\\n')
                f.write(f'**AI Prediction:**\\n')
                f.write(f'- Mechanism: {ai_pred.get(\"mechanism\", \"Unknown\")}\\n')
                f.write(f'- Probability: {ai_pred.get(\"probability\", \"Unknown\")}\\n')
                f.write(f'- Evidence: {ai_pred.get(\"evidence\", \"None provided\")}\\n\\n')
                
                f.write(f'**Validation Results:**\\n')
                val_summary = result.get('validation_summary', {})
                f.write(f'- Validated in {val_summary.get(\"validated_databases\", 0)}/{val_summary.get(\"total_databases\", 0)} databases\\n')
                f.write(f'- Confidence Level: {val_summary.get(\"confidence_level\", \"Unknown\")}\\n')
                f.write(f'- Evidence Strength: {result.get(\"evidence_strength\", \"Unknown\")}\\n\\n')
                
                f.write('---\\n\\n')
        
        if novel_count == 0:
            f.write('No novel interactions found - all predictions validated against known databases.\\n\\n')
        
        # Validated interactions section
        f.write('## Validated Interactions (Known)\\n\\n')
        validated_count = 0
        for result in results:
            if not result.get('novel_interaction', False):
                validated_count += 1
                gene_a = result['gene_pair'][0]
                gene_b = result['gene_pair'][1]
                val_summary = result.get('validation_summary', {})
                
                f.write(f'### {gene_a} - {gene_b}\\n\\n')
                f.write(f'- Confidence Level: {val_summary.get(\"confidence_level\", \"Unknown\")}\\n')
                f.write(f'- Validated in {val_summary.get(\"validated_databases\", 0)}/{val_summary.get(\"total_databases\", 0)} databases\\n\\n')
        
        if validated_count == 0:
            f.write('No known interactions found - all predictions appear to be novel.\\n\\n')
    
    print('Detailed report generated successfully')
    
except Exception as e:
    print(f'Error generating detailed report: {e}', file=sys.stderr)
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Detailed report generated: $DETAILED_REPORT"
else
    print_warning "Could not generate detailed report"
fi

# Generate validation statistics
print_step "Step 5: Analysis Complete"

print_info "Validation workflow completed successfully!"
echo ""
echo "Generated files in $OUTPUT_DIR:"

# List generated files
echo ""
echo "Core results:"
if [ -f "$VALIDATION_OUTPUT" ]; then
    echo "  ✓ $VALIDATION_OUTPUT (raw validation data)"
fi

echo ""
echo "Reports:"
if [ -f "$SUMMARY_REPORT" ]; then
    echo "  ✓ $SUMMARY_REPORT (summary statistics)"
fi
if [ -f "$DETAILED_REPORT" ]; then
    echo "  ✓ $DETAILED_REPORT (detailed analysis)"
fi

echo ""
echo "Database cache:"
if [ -d "$CACHE_DIR" ]; then
    CACHE_COUNT=$(find "$CACHE_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ $CACHE_DIR ($CACHE_COUNT cached database queries)"
fi

echo ""
echo "Quick Summary:"
if [ -f "$SUMMARY_REPORT" ]; then
    echo "----------------------------------------"
    head -20 "$SUMMARY_REPORT" | tail -15
    echo "----------------------------------------"
fi

echo ""
echo "Next steps:"
echo "  1. Review the summary report: $SUMMARY_REPORT"
echo "  2. Examine novel interactions in: $DETAILED_REPORT"  
echo "  3. Use validation data to prioritize experimental validation"
echo "  4. Cache will speed up future validations of the same gene pairs"
echo ""
echo "Validation workflow completed!"