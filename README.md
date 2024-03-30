# LLM-Report-Labeling
This is the accompanying repository for the Paper _Is open-source there yet? A comparative study on commercial and open-source LLMs in their ability to label Chest X-Ray reports_.

The repository contains the code that was used to perform the labeling of chest x-ray reports using GPT-4 and different open-source Large Language Models hosted that were locally hosted through vLLM. 

## Usage

Follow these steps to perform the local LLM inference:

1. Follow the Quickstart Instructions of VLLM to locally deploy a Open-AI compatible server. This server will be queried by the labeling script in this repository through the Open-AI Python API. 

2. Setup the CSV file that contains the free-text reports that should be labeled. Three columns: 'Patient MRN', 'Accession Number' and 'Report Text' are expected by default. The _prompt.py_ can be adjusted for individual use cases.

3. Run the _create_labels_vllm.py_ script. 

## Paper 
TBD 
