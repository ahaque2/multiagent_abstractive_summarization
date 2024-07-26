# About

The goal of this repository is to demonstrate how multiagent teams of LLM's can be used to perform abstractive summarisations of multiple documents containing multiple perspectives.

# Getting Started

1. Clone the repo
2. Create a venv `python3 -m venv venv` in the repo folder
3. `pip install -r requirements.txt`
4. `export LAS_API_TOKEN=<YOUR_TOKEN>`
5. Run `python3 scripts/multiperspective.py` to see a demonstration of the mutliagent debate and writer critic workflow applied to the conflicting news dataset.