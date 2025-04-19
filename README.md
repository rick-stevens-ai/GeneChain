 # Gene/Protein Interaction Chain Finder

 This project uses OpenAI GPT-4 to discover mechanistic interaction chains between genes or proteins. Each interaction in the chain is annotated with a subjective probability and visualized as a network graph.

 ## Features

 - Query GPT-4 to find causal interaction paths between two biological entities.
 - Output raw JSON, Graphviz DOT, and rendered PNG diagrams.
 - Batch processing from an input file of entity pairs.

 ## Requirements

 - Python ≥ 3.8
 - openai
 - networkx
 - matplotlib (optional, for rendering PNG)
 - pydot (for generating DOT files)
 - Graphviz (for rendering DOT files; install via your OS package manager)

 Install Python dependencies:

 ```bash
 pip install -r requirements.txt
 ```

 ## Usage

 Export your OpenAI API key:

 ```bash
 export OPENAI_API_KEY="sk-..."
 ```

 Single pair interaction chain discovery:

 ```bash
 python gene_chain_v1.py TP53 EGFR --model gpt-4o --paths 4
 ```

 Batch mode:

 ```bash
 python gene_chain_v1.py --input-file pairs.txt --model gpt-4o --paths 3 --out network
 ```

 ## Output

 - `<out>_interactions.json`: Raw JSON with interaction paths.
 - `<out>.dot`: Graphviz DOT file for the interaction network.
 - `<out>.png`: Rendered PNG diagram of the network.

 ## License

 Add license information here.