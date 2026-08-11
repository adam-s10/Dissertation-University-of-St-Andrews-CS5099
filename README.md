# Humans Are The Predictable Ones:

## Detecting Machine-Generated Text with Base Model Probability Curvature

---
_**CS5099 Dissertation in Computer Science**_

**School of Computer Science**

**MSc in Artificial Intelligence**

_Academic Year 2025–2026_

_Adam Steinmetz_

_250030052_

---

### Abstract
Large Language Models have seen continuous integration in society, workplaces, and academia for various tasks such as initial ideation and summarization. However, they also can be used maliciously, from plagiarism, academic integrity concerns, and deepfakes. The requirements to have responsible Artificial Intelligence and a reliable way to detect its use continue to grow. The zero-shot detector Fast-DetectGPT [1] provides a promising baseline to achieve these goals for machine-generated text: capable of doing so in a black-box setting, at scale, and at speed. This paper replicates the work of Bao et al. supporting their hypothesis that conditional probability curvature serves as an effective detection method for machine-generated text while contradicting machine probability curvature is greater than that of human curvature. Furthermore, this paper explores and verifies the hypothesis that smaller language models are better black-box detectors [2], integrating three base-models with sizes 0.5–3B parameters with the Fast-DetectGPT algorithm — identifying that the algorithm is constrained by a minimum word count and domain of the candidate passage.

---

[1] G. Bao, Y. Zhao, Z. Teng, L. Yang, and Y. Zhang, “Fast-detectgpt: Efficient zero-shot detection
    of machine-generated text via conditional probability curvature,” in International Conference on
    Learning Representations, 2024, pp. 24814–24836.

[2] N. Mireshghallah, J. Mattern, S. Gao, R. Shokri, and T. Berg-Kirkpatrick, “Smaller language
    models are better black-box machine-generated text detectors,” arXiv preprint arXiv:2305.09859, 2023.

---

---

---

To replicate the results of this project from scratch, large datasets must be downloaded from Hugging Face. As part of the submission, all datasets from after the sampling steps have been provided.


### Data
- Sampled Datasets
  - Sampled parquet files for each dataset after train-test split
- Generated Datasets
  - Collection of datasets after all six generation models were run to create positive samples of machine-generated text.
- Results Files
  - Files containing curvature returned by Fast-DetectGPT sorted by the generation models

### Models
List of models used with their corresponding string representations and Hugging Face Repository IDs (HF).
#### Generation Models

- Lily-Qwen1.5-0.5B
  - String representation: 'qwen_ft'
  - HF: kulia-moon/ Lily-Qwen1.5-0.5B
- Llama-3.2-1B-Instruct
  - String representation: 'llama'
  - HF: meta-llama/Llama-3.2-1B-Instruct
- gemma-2-2b-it
  - String representation: 'google'
  - HF: google/gemma-2-2b-it 
- rocket-3B 
  - String representation: 'rocket'
  - HF: pansophic/rocket-3B
- Llama-3.1-Nemotron-Nano-4B-v1.1
  - String representation: 'nvidia_4b'
  - HF: nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
- OpenReasoning-Nemotron-7B
  - String representation: 'nvidia_7b'
  - HF: nvidia/OpenReasoning-Nemotron-7B

#### Detection Models
- Qwen
  - String representation: 'qwen'
  - HF: Qwen/Qwen1.5-0.5B
- Bitnet
  - String representation: 'bitnet'
  - HF: microsoft/bitnet-b1.58-2B-4T-bf16
- Stable LM
  - String representation: 'stablelm'
  - HF: stabilityai/stablelm-3b-4e1t

---

### Usage

**Usage notice: `download_models.py` and `download_data.py` make use of the Hugging Face Cli and expect an active Hugging Face token in a .env file in the format HF_TOKEN = '*your token here*'. You will need to add the .env file to the top directory of this project and acquire the token yourself**

#### Requirements
Firstly install the requirements by running:
```bash 
pip install -r requirements.txt
```

#### Download Models
A script which will download a default, specific, or all models used in this project executed as:
`download_models.py [--all] [--model]`. 

Where: 
- `[--all]` will download all models. Defaults to `True` 
- `[--model]` is the string representation of one a single model to be downloaded. Defaults to `'qwen'`


Example usage:
```bash
python download_models.py --all True
```

#### Download Datasets
A script to download the raw datasets required for this project.

Example usage:
```bash
python download_data.py
```

#### Sample Datasets
A script to sample human-written examples from the raw datasets used in this project executed as `sample_datasets [--delete-dir] [--num-samples]`. 

Where:
- `[--delete-dir]` deletes the raw dataset directory. Defaults to `False`
- `[--num-samples]` is the number of samples taken. Defaults to `1000`

Applies train-test split with 80% train and 20% test.

Example usage:

```bash
python sample_datasets.py
```

#### Process Datasets
A script to process the datasets, creating curated prompts depending on the dataset executed as `process_datasets.py`

Usage:
```bash
python process_datasets.py
```

#### Generate Datasets
A script to generate examples of machine-generated text on a per-dataset basis executed as `generate_datasets.py <--model> [--batch-size]`. 

Where:
- `<--model>` is the string representation of the generation model
- `[--batch-size]` is number of lines processed in the dataset at a time. Defaults to `20`

*Note: this script must be manually run for each train-test split dataset and generation model combination.*

Example usage:
```bash 
python generate_datasets.py --model nvidia_7b --batch-size 10
```

#### Fast-DetectGPT
a script to run Fast-DetectGPT on the provided dataset, appending the sampling model, scoring model, curvature, and runtime to already existing columns extracted from the dataset passed executed as `fast_detect_gpt.py <--sampling-model> <--dataset> <--save-file> [--scoring-model] [--save-dir]`. 

*Note: this script must be manually run for each dataset, sampling model, and score model combination you wish to use.*

Where: 
- `<--sampling-model>` the sampling model to be used 
- `<--dataset>` the name of the dataset to run the algorithm on
- `<--save-file>` name of the results file. Must be saved as a .csv or .parquet
- `[--scoring-model]` the scoring model to be used. Defaults to `None` (same as the sampling model)
- `[--save-dir]` the directory to save the results in. Defaults to `'Results Files'`

Example usage:
```bash
python fast_detect_gpt.py --sampling-model qwen --dataset google_test_pubmedqa_generated.parquet --save-file qwen_sampled_and_scored_on_google_test_pubmedqa_generated.csv
```

--- 

---

---

### GitHub Repo

_Note that in line with University Guidelines, the repo has been set to private. In the event you wish to view it, please email as760@st-andrews.ac.uk and I will give you access._

GitHub: https://github.com/adam-s10/Dissertation-University-of-St-Andrews-CS5099