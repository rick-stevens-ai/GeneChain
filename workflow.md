# Gene Chain Analysis Workflow

This document explains how the Python scripts in this gene chain analysis project relate to each other and can be used together to analyze gene interactions and pathways.

## Project Overview

This is a bioinformatics project that uses AI (primarily GPT models) to discover, analyze, and compare gene/protein interaction pathways. The workflow supports both individual gene pair analysis and batch processing of multiple pairs.

## Core Components

### 1. Configuration and Model Management

**`model_config.py`** - Centralized model server configuration
- Loads configuration from `model_servers.yaml` (or JSON fallback)
- Handles environment variable substitution for API keys
- Provides functions to get model configurations by name or shortname
- Supports multiple AI model endpoints (OpenAI, local servers, etc.)

### 2. Data Generation Pipeline

**`gene_chain_v1.py`** - Core interaction discovery engine
- **Purpose**: Queries AI models to discover mechanistic interaction pathways between two genes/proteins
- **Input**: Gene/protein names (single pair or batch from file)
- **Output**: JSON files with interaction paths, DOT files for visualization, PNG network diagrams
- **Features**: Checks for existing analyses to avoid redundant API calls, caches results in `INTERACTION_CACHE/`

**`context_to_pairs.py`** - Extract gene pairs from biological text
- **Purpose**: Extracts gene/protein entities from unstructured biological text and generates gene pairs
- **Workflow**: Text → entity extraction → pair generation → calls `gene_chain_v1.py`
- **Features**: Skips pairs that already have existing interaction data

**`random_gene_pairs.py`** - Generate random gene pairs for analysis
- **Purpose**: Creates random gene pairs from a list/CSV file for systematic analysis
- **Input**: CSV/TSV file with gene names
- **Output**: Text file with gene pairs suitable for batch processing

### 3. Analysis and Reporting

**`gene_pathway_report.py`** - Comprehensive pathway analysis reports
- **Purpose**: Generates detailed markdown reports analyzing interaction pathways
- **Features**: 
  - Classifies interactions as well-supported vs. conjectural based on evidence
  - Checks cache first, generates new data if needed via `gene_chain_v1.py`
  - Supports both single pairs and batch processing
- **Output**: Detailed markdown reports with pathway classifications

**`interactions_to_summary.py`** - Generate biological summaries
- **Purpose**: Creates comprehensive biological summaries from interaction JSON files
- **Uses**: AI to interpret pathway data and provide biological context
- **Output**: Text summaries explaining the biological significance of interactions

### 4. Data Processing and Extraction

**`extract_pathways.py`** - Extract structured data from markdown reports
- **Purpose**: Parses markdown pathway reports and extracts structured data
- **Output Formats**: JSON, CSV, TSV, or markdown
- **Features**: Configurable evidence inclusion, pathway limits

**`extract_secondary_genes.py`** - Analyze intermediate genes in pathways
- **Purpose**: Identifies and analyzes secondary (intermediate) genes that appear in interaction pathways
- **Features**: 
  - Frequency analysis of secondary genes
  - Identifies tentative interactions (containing uncertainty language)
  - Supports filtering by occurrence frequency

**`context_to_narrative.py`** - Generate structured narratives from biological text
- **Purpose**: Converts unstructured biological context into structured narratives
- **Output**: Detailed explanations with entities, interactions, effects, and references

### 5. Comparative Analysis

**`compare_model_predictions.py`** - Compare predictions across multiple AI models
- **Purpose**: Analyzes how different AI models predict interactions for the same gene pairs
- **Features**:
  - Tests model endpoint availability
  - Generates predictions from multiple models
  - Performs traditional comparison and GPT-4.1 meta-analysis
  - Provides both statistical and narrative comparisons

**`compare_pathways.py`** - Compare pathway analyses across multiple files
- **Purpose**: Identifies common and unique pathways across different analysis files
- **Input**: Multiple `*-TF_pathways.json` files
- **Output**: Categorized pathways (common to all, specific subsets, unique)

**`extract_comparative_analysis.py`** - Extract and format comparative analysis results
- **Purpose**: Processes comparison reports and formats GPT-4.1 comparative analyses
- **Output Formats**: Text, markdown, or HTML with formatted comparative insights

## Common Workflows

### Workflow 1: Single Gene Pair Analysis
```
1. gene_chain_v1.py TP53 EGFR → Generate basic interaction data
2. gene_pathway_report.py TP53 EGFR → Generate detailed analysis report
3. interactions_to_summary.py → Generate biological summary
```

### Workflow 2: Batch Analysis from Text
```
1. context_to_pairs.py -i biological_text.txt → Extract entities and generate pairs
2. gene_pathway_report.py --input-file pairs.txt → Generate reports for all pairs
3. extract_pathways.py report.md → Extract structured data
4. extract_secondary_genes.py → Analyze intermediate genes
```

### Workflow 3: Random Gene Set Analysis
```
1. random_gene_pairs.py --input-file genes.csv --num-pairs 20 → Generate random pairs
2. gene_chain_v1.py --input-file pairs.txt → Process all pairs
3. compare_pathways.py → Compare results across pairs
```

### Workflow 4: Model Comparison Study
```
1. compare_model_predictions.py --pairs 10 --models gpt41 claude → Compare multiple models
2. extract_comparative_analysis.py → Format comparative analysis results
```

## Data Flow and Dependencies

### Primary Dependencies:
- **model_config.py** ← Used by most scripts for AI model configuration
- **gene_chain_v1.py** ← Core engine used by `context_to_pairs.py` and `gene_pathway_report.py`

### Data Files:
- **INTERACTION_CACHE/** - Centralized storage for all interaction data
- **model_servers.yaml** - Model configuration file
- **pairs.txt** - Standard format for gene pair lists
- **\*_interactions.json** - Raw interaction data from AI models
- **\*_pathways.json** - Structured pathway data for comparison
- **\*.md** - Markdown reports with detailed analysis

### File Naming Conventions:
- `network_GENE1_GENE2_interactions.json` - Raw interaction data
- `GENE1_GENE2_pathways.json` - Processed pathway data  
- `gene_pathway_report_GENE1_GENE2.md` - Individual reports
- `summary_GENE1_GENE2.txt` - Biological summaries

## Configuration

The system uses `model_servers.yaml` to configure different AI model endpoints:
- Supports OpenAI, local servers, or other API-compatible endpoints
- Environment variable substitution for API keys
- Model shortnames for easier reference

## Key Features

1. **Caching**: Automatic result caching to avoid redundant API calls
2. **Batch Processing**: Support for processing multiple gene pairs efficiently  
3. **Multi-model Support**: Compare results across different AI models
4. **Evidence Classification**: Distinguishes well-supported from conjectural interactions
5. **Format Flexibility**: Multiple output formats (JSON, CSV, markdown, HTML)
6. **Error Handling**: Graceful handling of missing data and API failures

This workflow enables comprehensive analysis of gene interaction networks using AI, from initial discovery through comparative analysis and detailed reporting.