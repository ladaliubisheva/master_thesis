# EXPERIMENTAL EVALUATION OF TOKENIZATION STRATEGIES FOR LARGE LANGUAGE MODELS IN FINANCIAL TIME SERIES FORECASTING 

**Type:** Master's Thesis

**Author:** Lada Liubisheva

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Jan Mendling

![pipeline](/pipeline.png)

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

Application of Large Language Models (LLMs) to time series forecasting has grown at a rapid rate, yet their success is highly dependent on the way continuous numerical data is converted into discrete token sequences—a design decision that has not been sufficiently investigated. This thesis explores the effect of various tokenization techniques on the accuracy of LLMs' prediction in the context of univariate financial time series with daily closing prices of the S&P 500 as a specific example case. The study considers three tokenization techniques: digit-space encoding (LLMTime), distributional quantization (Gaussian Binning), and learned discrete representations through Vector Quantized Variational Autoencoders (VQ-VAE). These are applied under one GPT-2-based autoregressive forecasting framework and compared across different horizons using RMSE, MAPE, and directional accuracy metrics. 
Results highlight key trade-offs: VQ-VAE achieves improved numerical accuracy but fails to capture directional trend on longer time horizons; Gaussian Binning attains relatively stable performance at the expense of losing granularity and accuracy; and LLMTime addresses interpretability with reliable trend detection. The findings highlight that tokenization is not just a preprocessing step but a fundamental modeling decision with profound implications for forecast results. This work provides practical insights for selecting tokenization approaches in LLM-driven financial time series forecasting. 


**Keywords**: tokenization strategies, time series forecasting, Large Language Models (LLMs), financial prediction, vector quantization


## Working with the repo

### Python Version
This project is designed for **Google Colab environment** (Python 3.10+)

The code is optimized for Google Colab and includes all necessary dependency installations in the first cells:
```python
# CELL 1: Install Dependencies (Run Once)
!pip install openpyxl
```

All other required packages (PyTorch, Transformers, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib) are pre-installed in the Google Colab environment.
Key Dependencies:

OpenPyXL: Excel file handling (installed via pip in first cell)
PyTorch: Pre-installed in Colab for neural network operations
Transformers: Pre-installed in Colab for GPT-2 implementation
NumPy/Pandas: Pre-installed in Colab for data processing
Scikit-learn: Pre-installed in Colab for evaluation metrics
SciPy: Pre-installed in Colab for statistical functions

For local execution, install:
```python
pip install torch transformers tokenizers datasets
pip install pandas numpy scikit-learn matplotlib scipy openpyxl
```
### Setup

1. Open the notebook in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hug5-Zz2sMHq0vONGFTWEfJQ2jMareCa?usp=sharing)

2. Run Cell 1 to install additional dependencies
3. Upload raw_data.csv file when prompted in Cell 2
4. Execute cells sequentially

Hardware Requirements:

Google Colab Free: Sufficient for experimentation
Google Colab Pro: Recommended for faster execution 
The code automatically detects and uses GPU when available in Colab

## Reproducing results
All experiments are fully reproducible through deterministic seeding and controlled experimental conditions. The notebook implements a comprehensive evaluation framework that ensures fair comparison across tokenization methods.

### Training code

The repository contains complete training pipelines for:
- **VQ-VAE models**: Neural vector quantization training with encoder/decoder networks
- **GPT-2 models**: Transformer training with method-specific tokenized sequences

**Non-training components:**
- **LLMTime**: Numerical serialization (configuration-based, no training required)
- **Gaussian Binning**: Statistical tokenizer initialization (configuration-based, no training required)

All training procedures use identical hyperparameters and architectures, differing only in tokenization approach.

### Evaluation code
The repository implements a rigorous evaluation framework designed for fair comparison across tokenization methods.

**Evaluation Metrics:**
- **RMSE**: Root Mean Square Error 
- **MAPE**: Mean Absolute Percentage Error  
- **Directional Accuracy**

**Evaluation Design:**
- **Multi-horizon forecasting**: 1, 5, and 10-day prediction horizons
- **Multiple test periods**: 5 distinct evaluation windows per horizon
- **Temporal split**: 2022 cutoff ensuring no data leakage
- **Controlled generation**: Identical parameters (temperature=0.9, top_k=50) across all methods

**Robustness Framework:**
- **Multiple random seeds**: All experiments repeated across 3 fixed random seeds
- **Independent runs**: Complete model retraining for each seed
- **Statistical aggregation**: Mean and standard deviation reported across runs
- **Fair comparison**: Same computational budget and evaluation conditions for all methods

The evaluation framework ensures that performance differences reflect the underlying tokenization strategy rather than random initialization variance or experimental design choices.

### Pretrained models

No pretrained models provided - all models trained from scratch for experimental consistency.

## Results
The comparative analysis results are stored in the repository as `results.png`, which displays the performance comparison across all three tokenization methods (LLMTime, Gaussian Binning, and VQ-VAE) for different prediction horizons.

The complete experimental code and detailed results can be accessed through the Google Colab notebook linked above. The notebook automatically generates:
- Method rankings by evaluation metrics
- Individual run breakdowns showing consistency across random seeds
For the full experimental results and analysis, please refer to the interactive notebook.

## Project structure

```bash
├── README.md
├── raw_data.csv                                    -- stores data file 
├── results.png                                     -- stores results table
├── pipeline.png                                    -- stores approach illustration              
```
