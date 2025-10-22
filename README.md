# Situating and Comparing Artificial Neural Networks in a Unified Brain-like Space

## Overview
This repository provides code for computing graph metrics from spatial attention patterns across different models and brain functional networks.

## Repository Structure

### 1. Model Processing (`model-process/`)
Contains code for computing graph metrics from spatial attention patterns of various model families.

#### Key Files:
### Vision Transformers (ViT, ViT-Variants)
  - `vit-process.py`
  - `dinov3-process.py`

### Large Language Models (LLMs)
  - `gpt2-process.py`
  - `bert-process.py`

### Large Multimodal Models (LMMs)
  - `clip-language.py`, `clip-vision.py`
  - `blip-language.py`, `blip-vision.py`

### RoPE-based Models （LLM-RoPEs、LMM-RoPEs）
- **LLMs with Rotary Position Embeddings**
  - `qwen-process.py`
- **LMMs with Rotary Position Embeddings**
  - `qwen-language-process.py`, `qwen-vision-process.py`
  - `deepseek-language-process.py`

#### Usage Requirements:
Before running any model processing scripts, modify these key parameters:
1. **Model List**: Specify target list of model names in `models`
2. **Cache Directory**: Set `your_cache_dir` for model downloads
3. **Output Directory**: Set `your_base_dir` for results storage
4. **Access Tokens**: For gated models, add your HuggingFace access token in the download section (request from official model pages)

#### Extension Notes:
For other and new RoPE-based LLM/LMM models, adapt existing examples by:
- Modifying the model loader
- Adjusting configuration extraction based on model architecture
- Updating Q/K weight extraction methods according to model nesting structure

### 2. Brain Network Processing (`brain-process.ipynb`)
Computes graph metrics for brain functional networks.

#### Prerequisites:
- Prepare multiple pre-extracted brain network connectivity matrices (`.mat` files)
- Ensure main diagonal elements are set to zero in all matrices

### 3. Pre-computed Results
- `model-result.zip`: Graph metrics for 151 models used in the paper, please extract as'model result/'
- `brainnetwork-result.json`: Graph metrics for 7 group-level brain functional networks

### 4. Brain-Like Space Analysis (`brain-like_space.ipynb`)
Constructs the brain-like space and computes brain-like scores.

#### Quick Start:
Use the provided `model-result.zip` and `brainnetwork-result.json` directories to:
- Visualize the brain-like space
- Generate brain-like score result files

## Requirements

### Core Dependencies
- torch==2.8.0
- transformers==4.56.1
- accelerate==1.10.1
- safetensors==0.6.2
- timm==1.0.14
- networkx==3.4.2
- python-louvain==0.16
- numpy==2.2.6
- scipy==1.15.3

**Note:** `transformers` and `accelerate` versions may need adjustment for specific models. Some models require different versions for compatibility.

## Citation
If you use this code in your research, please cite our accompanying paper：Situating and Comparing Artificial Neural Networks in a Unified Brain-like Space
