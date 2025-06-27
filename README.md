# Title

**Type:** Master's Thesis

**Author:** Lada Liubisheva

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Jan Mendling

![results](/approach.png)

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

The use of Large Language Models (LLMs) for time series prediction has been increasingly growing. However, their performance is strongly dependent on how to convert continuous numerical values to discrete token sequences—a design decision under-explored in previous works. This thesis explores the effect of various tokenization methods on predictive performance of LLMs, within the setting of univariate financial time series prediction,  with S&P 500 daily closing prices as a concrete example. Three methods of tokenization are considered: digit-space encoding (LLMTime), distributional quantization (Gaussian Binning), and learned latent codes from Vector Quantized Variational Autoencoders (VQ-VAE). They are integrated in a common GPT-2-based autoregressive prediction model and compared in terms of RMSE(Root Mean Square Error), MAPE(Mean Absolute Percentage Error), and directional accuracy across multiple prediction horizons.
The results reveal important trade-offs. Whereas VQ-VAE exhibits greater numerical accuracy, it fails to correctly detect directional trends at long horizons. Gaussian Binning provides time-resistant performance at the expense of granularity and accuracy. LLMTime provides the most well-rounded results by merging interpretability and consistent trend detection.
In short, the study highlights that tokenization is not just a preprocessing step, but instead an inherent modeling decision with significant impact on downstream forecasting performance. These results offer practical guidance for selecting tokenization methods in LLM-based financial time series applications.


**Keywords**: tokenization strategies, time series forecasting, Large Language Models (LLMs), financial prediction, vector quantization

**Full text**: [include a link that points to the full text of your thesis]
*Remark*: a thesis is about research. We believe in the [open science](https://en.wikipedia.org/wiki/Open_science) paradigm. Research results should be available to the public. Therefore, we expect dissertations to be shared publicly. Preferably, you publish your thesis via the [edoc-server of the Humboldt-Universität zu Berlin](https://edoc-info.hu-berlin.de/de/publizieren/andere). However, other sharing options, which ensure permanent availability, are also possible. <br> Exceptions from the default to share the full text of a thesis require the approval of the thesis supervisor.  

## Working with the repo

### Python Version
This project is designed for **Google Colab environment** (Python 3.10+)

The code is optimized for Google Colab and includes all necessary dependency installations in the first cells:
```python
# CELL 1: Install Dependencies (Run Once)
!pip install openpyxl
```
### Setup

[This is an example]

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducing results

Describe steps how to reproduce your results.

Here are some examples:
- [Paperswithcode](https://github.com/paperswithcode/releasing-research-code)
- [ML Reproducibility Checklist](https://ai.facebook.com/blog/how-the-ai-community-can-get-serious-about-reproducibility/)
- [Simple & clear Example from Paperswithcode](https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md) (!)
- [Example TensorFlow](https://github.com/NVlabs/selfsupervised-denoising)

### Training code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

### Pretrained models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project structure

(Here is an example from SMART_HOME_N_ENERGY, [Appliance Level Load Prediction](https://github.com/Humboldt-WI/dissertations/tree/main/SMART_HOME_N_ENERGY/Appliance%20Level%20Load%20Prediction) dissertation)

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file 
├── plots                                           -- stores image files
└── src
    ├── prepare_source_data.ipynb                   -- preprocesses data
    ├── data_preparation.ipynb                      -- preparing datasets
    ├── model_tuning.ipynb                          -- tuning functions
    └── run_experiment.ipynb                        -- run experiments 
    └── plots                                       -- plotting functions                 
```
