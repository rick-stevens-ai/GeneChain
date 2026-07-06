#!/bin/bash
#
# Gene Chain Analysis Utilities
# ==============================
# 
# This script provides common utility functions for managing and 
# working with the gene chain analysis pipeline.
#
# Usage: ./gene_chain_utils.sh COMMAND [ARGS...]
#
# Commands:
#   setup              Set up the environment and check dependencies
#   clean              Clean up temporary files and caches
#   status             Show status of analysis files and cache
#   list-models        List available AI models
#   test-models        Test AI model endpoints
#   backup-cache       Backup the INTERACTION_CACHE directory
#   restore-cache      Restore the INTERACTION_CACHE from backup
#   find-interactions  Find interaction files for specific gene pairs
#   summarize-results  Generate a summary of all analysis results
#   help               Show detailed help for each command
#

set -e  # Exit on any error

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="$SCRIPT_DIR/INTERACTION_CACHE"
BACKUP_DIR="$SCRIPT_DIR/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup environment and check dependencies
setup_environment() {
    print_info "Setting up Gene Chain Analysis environment..."
    echo ""
    
    print_info "Checking Python dependencies..."
    local missing_deps=()
    
    # Check for Python
    if ! command_exists python && ! command_exists python3; then
        missing_deps+=("python")
    fi
    
    # Check for required Python packages
    local python_cmd="python3"
    if ! command_exists python3; then
        python_cmd="python"
    fi
    
    local required_packages=("openai" "networkx" "matplotlib" "pyyaml")
    for package in "${required_packages[@]}"; do
        if ! $python_cmd -c "import $package" 2>/dev/null; then
            missing_deps+=("python-$package")
        fi
    done
    
    # Check for optional but recommended tools
    print_info "Checking optional tools..."
    if ! command_exists jq; then
        print_warning "jq not found (recommended for JSON processing)"
        echo "  Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
    else
        print_success "jq found"
    fi
    
    if ! command_exists dot; then
        print_warning "Graphviz not found (recommended for network visualization)"
        echo "  Install with: brew install graphviz (macOS) or apt-get install graphviz (Ubuntu)"
    else
        print_success "Graphviz found"
    fi
    
    # Report missing dependencies
    if [ ${#missing_deps[@]} -eq 0 ]; then
        print_success "All core dependencies are installed"
    else
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo "Install missing Python packages with:"
        echo "  pip install openai networkx matplotlib pyyaml"
        return 1
    fi
    
    # Check environment variables
    print_info "Checking environment variables..."
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY not set"
        echo "  Set with: export OPENAI_API_KEY='your-api-key'"
    else
        print_success "OPENAI_API_KEY is set"
    fi
    
    # Create necessary directories
    print_info "Creating necessary directories..."
    mkdir -p "$CACHE_DIR"
    mkdir -p "$BACKUP_DIR"
    print_success "Directories created"
    
    print_success "Environment setup completed"
}

# Clean up temporary files and caches
clean_environment() {
    print_info "Cleaning up temporary files..."
    
    # Ask for confirmation
    echo "This will remove:"
    echo "  - Temporary analysis directories (batch_analysis_*, random_gene_analysis_*, model_comparison_*)"
    echo "  - Partial result files (*.partial)"
    echo "  - Temporary pairs files (pairs.txt if in current directory)"
    echo ""
    echo "INTERACTION_CACHE will be preserved unless you specify --cache"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleanup cancelled"
        return 0
    fi
    
    # Remove temporary directories
    print_info "Removing temporary analysis directories..."
    rm -rf batch_analysis_* random_gene_analysis_* model_comparison_* 2>/dev/null || true
    
    # Remove partial files
    print_info "Removing partial result files..."
    rm -f *.partial 2>/dev/null || true
    
    # Remove temporary pairs file if it exists in current directory
    if [ -f "pairs.txt" ]; then
        print_info "Removing temporary pairs.txt..."
        rm -f pairs.txt
    fi
    
    # Clean cache if requested
    if [ "$1" = "--cache" ]; then
        print_warning "Removing INTERACTION_CACHE directory..."
        rm -rf "$CACHE_DIR"
        mkdir -p "$CACHE_DIR"
    fi
    
    print_success "Cleanup completed"
}

# Show status of analysis files and cache
show_status() {
    print_info "Gene Chain Analysis Status"
    echo "=========================="
    echo ""
    
    # Cache status
    print_info "Cache Status:"
    if [ -d "$CACHE_DIR" ]; then
        local interaction_count=$(find "$CACHE_DIR" -name "*interactions.json" 2>/dev/null | wc -l | tr -d ' ')
        local dot_count=$(find "$CACHE_DIR" -name "*.dot" 2>/dev/null | wc -l | tr -d ' ')
        local png_count=$(find "$CACHE_DIR" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        local summary_count=$(find "$CACHE_DIR" -name "summary_*.txt" 2>/dev/null | wc -l | tr -d ' ')
        
        echo "  Interaction files: $interaction_count"
        echo "  Network diagrams (DOT): $dot_count"
        echo "  Network images (PNG): $png_count"
        echo "  Biological summaries: $summary_count"
        echo "  Cache directory: $CACHE_DIR"
    else
        echo "  Cache directory not found"
    fi
    echo ""
    
    # Recent analysis directories
    print_info "Recent Analysis Runs:"
    local recent_dirs=($(find . -maxdepth 1 -type d -name "*analysis_*" -o -name "*comparison_*" 2>/dev/null | sort -r | head -5))
    if [ ${#recent_dirs[@]} -gt 0 ]; then
        for dir in "${recent_dirs[@]}"; do
            echo "  $dir ($(ls "$dir" 2>/dev/null | wc -l | tr -d ' ') files)"
        done
    else
        echo "  No recent analysis directories found"
    fi
    echo ""
    
    # Configuration status
    print_info "Configuration:"
    if [ -f "model_servers.yaml" ]; then
        echo "  ✓ model_servers.yaml found"
    elif [ -f "model_servers.json" ]; then
        echo "  ✓ model_servers.json found"
    else
        echo "  ⚠ No model configuration file found"
    fi
    
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "  ✓ OPENAI_API_KEY is set"
    else
        echo "  ⚠ OPENAI_API_KEY not set"
    fi
    echo ""
    
    # Available workflow scripts
    print_info "Available Workflows:"
    local workflows=("workflow_single_pair.sh" "workflow_batch_from_text.sh" "workflow_random_gene_set.sh" "workflow_model_comparison.sh")
    for workflow in "${workflows[@]}"; do
        if [ -f "$workflow" ] && [ -x "$workflow" ]; then
            echo "  ✓ $workflow"
        else
            echo "  ⚠ $workflow (not found or not executable)"
        fi
    done
}

# List available AI models
list_models() {
    print_info "Available AI Models:"
    if command_exists python3; then
        python3 model_config.py --list 2>/dev/null || print_error "Could not load model configuration"
    elif command_exists python; then
        python model_config.py --list 2>/dev/null || print_error "Could not load model configuration"
    else
        print_error "Python not found"
    fi
}

# Test AI model endpoints
test_models() {
    print_info "Testing AI model endpoints..."
    local models=("gpt41" "claude")
    
    if [ $# -gt 0 ]; then
        models=("$@")
    fi
    
    for model in "${models[@]}"; do
        print_info "Testing model: $model"
        # This would need to be implemented in the Python scripts
        echo "  (Endpoint testing functionality to be implemented)"
    done
}

# Backup the INTERACTION_CACHE directory
backup_cache() {
    if [ ! -d "$CACHE_DIR" ]; then
        print_error "Cache directory not found: $CACHE_DIR"
        return 1
    fi
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="$BACKUP_DIR/interaction_cache_backup_$timestamp.tar.gz"
    
    print_info "Creating backup of interaction cache..."
    mkdir -p "$BACKUP_DIR"
    tar -czf "$backup_file" -C "$(dirname "$CACHE_DIR")" "$(basename "$CACHE_DIR")"
    
    if [ $? -eq 0 ]; then
        print_success "Backup created: $backup_file"
        
        # Show backup size
        local size=$(du -h "$backup_file" | cut -f1)
        echo "  Backup size: $size"
        
        # List contents summary
        local file_count=$(tar -tzf "$backup_file" | wc -l | tr -d ' ')
        echo "  Files backed up: $file_count"
    else
        print_error "Backup failed"
        return 1
    fi
}

# Restore the INTERACTION_CACHE from backup
restore_cache() {
    if [ $# -eq 0 ]; then
        print_info "Available backups:"
        local backups=($(find "$BACKUP_DIR" -name "interaction_cache_backup_*.tar.gz" 2>/dev/null | sort -r))
        if [ ${#backups[@]} -eq 0 ]; then
            print_warning "No backups found in $BACKUP_DIR"
            return 1
        fi
        
        for i in "${!backups[@]}"; do
            local backup="${backups[$i]}"
            local size=$(du -h "$backup" | cut -f1)
            local date=$(basename "$backup" | sed 's/interaction_cache_backup_\(.*\)\.tar\.gz/\1/')
            echo "  $((i+1)). $backup ($size, $date)"
        done
        
        echo ""
        read -p "Select backup number to restore (1-${#backups[@]}): " -r backup_num
        
        if [[ "$backup_num" =~ ^[0-9]+$ ]] && [ "$backup_num" -ge 1 ] && [ "$backup_num" -le ${#backups[@]} ]; then
            local selected_backup="${backups[$((backup_num-1))]}"
            restore_cache "$selected_backup"
        else
            print_error "Invalid selection"
            return 1
        fi
        return
    fi
    
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        return 1
    fi
    
    print_warning "This will replace the current cache directory"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Restore cancelled"
        return 0
    fi
    
    print_info "Restoring cache from backup..."
    
    # Remove current cache and restore from backup
    rm -rf "$CACHE_DIR"
    tar -xzf "$backup_file" -C "$(dirname "$CACHE_DIR")"
    
    if [ $? -eq 0 ]; then
        print_success "Cache restored from: $backup_file"
        
        # Show restore summary
        local file_count=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "  Files restored: $file_count"
    else
        print_error "Restore failed"
        return 1
    fi
}

# Find interaction files for specific gene pairs
find_interactions() {
    if [ $# -eq 0 ]; then
        print_info "Usage: $0 find-interactions GENE1 [GENE2]"
        print_info "  If GENE2 is not provided, finds all interactions involving GENE1"
        return 1
    fi
    
    local gene1="$1"
    local gene2="$2"
    
    print_info "Searching for interaction files..."
    
    if [ -n "$gene2" ]; then
        print_info "Looking for interactions between $gene1 and $gene2:"
        
        # Search for specific gene pair
        local patterns=(
            "*${gene1}*${gene2}*interactions*.json"
            "*${gene2}*${gene1}*interactions*.json"
        )
        
        for pattern in "${patterns[@]}"; do
            find . "$CACHE_DIR" -name "$pattern" 2>/dev/null
        done
    else
        print_info "Looking for all interactions involving $gene1:"
        
        # Search for any interactions involving gene1
        find . "$CACHE_DIR" -name "*${gene1}*interactions*.json" 2>/dev/null
    fi
}

# Generate a summary of all analysis results
summarize_results() {
    print_info "Generating comprehensive results summary..."
    
    local summary_file="gene_chain_results_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "Gene Chain Analysis Results Summary"
        echo "==================================="
        echo "Generated on: $(date)"
        echo ""
        
        echo "Environment Information:"
        echo "  Working directory: $(pwd)"
        echo "  Cache directory: $CACHE_DIR"
        echo "  Python version: $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo "Not found")"
        echo ""
        
        echo "Cache Statistics:"
        if [ -d "$CACHE_DIR" ]; then
            echo "  Interaction files: $(find "$CACHE_DIR" -name "*interactions.json" 2>/dev/null | wc -l | tr -d ' ')"
            echo "  Network diagrams: $(find "$CACHE_DIR" -name "*.dot" 2>/dev/null | wc -l | tr -d ' ')"
            echo "  Network images: $(find "$CACHE_DIR" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')"
            echo "  Biological summaries: $(find "$CACHE_DIR" -name "summary_*.txt" 2>/dev/null | wc -l | tr -d ' ')"
            echo "  Cache size: $(du -h "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "Unknown")"
        else
            echo "  Cache directory not found"
        fi
        echo ""
        
        echo "Analysis Runs:"
        local analysis_dirs=($(find . -maxdepth 1 -type d -name "*analysis_*" -o -name "*comparison_*" 2>/dev/null | sort -r))
        if [ ${#analysis_dirs[@]} -gt 0 ]; then
            for dir in "${analysis_dirs[@]}"; do
                local file_count=$(find "$dir" -type f 2>/dev/null | wc -l | tr -d ' ')
                local size=$(du -h "$dir" 2>/dev/null | cut -f1 || echo "Unknown")
                echo "  $dir: $file_count files, $size"
            done
        else
            echo "  No analysis directories found"
        fi
        echo ""
        
        echo "Recent Files Generated:"
        find . -name "*.md" -o -name "*pathway*.json" -o -name "*summary*.txt" -newer "$CACHE_DIR" 2>/dev/null | sort -r | head -10
        echo ""
        
        echo "Configuration Status:"
        if [ -f "model_servers.yaml" ] || [ -f "model_servers.json" ]; then
            echo "  ✓ Model configuration found"
        else
            echo "  ⚠ No model configuration found"
        fi
        
        if [ -n "$OPENAI_API_KEY" ]; then
            echo "  ✓ API key configured"
        else
            echo "  ⚠ API key not set"
        fi
        
    } > "$summary_file"
    
    print_success "Results summary generated: $summary_file"
    
    # Display the summary
    cat "$summary_file"
}

# Show detailed help
show_help() {
    echo "Gene Chain Analysis Utilities"
    echo "============================="
    echo ""
    echo "Usage: $0 COMMAND [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  setup              Set up environment and check dependencies"
    echo "  clean [--cache]    Clean temporary files (use --cache to also clear cache)"
    echo "  status             Show status of analysis files and cache"
    echo "  list-models        List available AI models from configuration"
    echo "  test-models [M...] Test AI model endpoints (optional model list)"
    echo "  backup-cache       Create a timestamped backup of INTERACTION_CACHE"
    echo "  restore-cache [F]  Restore cache from backup (interactive or specify file)"
    echo "  find-interactions GENE1 [GENE2]  Find interaction files for gene pair(s)"
    echo "  summarize-results  Generate comprehensive summary of all results"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                          # Initial environment setup"
    echo "  $0 status                         # Check current status"
    echo "  $0 clean                          # Clean temporary files"
    echo "  $0 backup-cache                   # Backup interaction cache"
    echo "  $0 find-interactions TP53 EGFR    # Find TP53-EGFR interactions"
    echo "  $0 find-interactions TP53         # Find all TP53 interactions"
    echo "  $0 summarize-results              # Generate comprehensive summary"
    echo ""
    echo "Workflow Scripts:"
    echo "  ./workflow_single_pair.sh         # Analyze single gene pair"
    echo "  ./workflow_batch_from_text.sh     # Batch analysis from text"
    echo "  ./workflow_random_gene_set.sh     # Random gene set analysis"
    echo "  ./workflow_model_comparison.sh    # Compare multiple AI models"
}

# Main command dispatcher
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        setup)
            setup_environment "$@"
            ;;
        clean)
            clean_environment "$@"
            ;;
        status)
            show_status "$@"
            ;;
        list-models)
            list_models "$@"
            ;;
        test-models)
            test_models "$@"
            ;;
        backup-cache)
            backup_cache "$@"
            ;;
        restore-cache)
            restore_cache "$@"
            ;;
        find-interactions)
            find_interactions "$@"
            ;;
        summarize-results)
            summarize_results "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"