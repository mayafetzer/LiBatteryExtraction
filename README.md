# MetalLeaching

By: Maya Fetzer, Jude Okolie, Alnur Gazizuly

Department of Chemical Engineering

Bucknell University

Machine learning framework for modeling and optimizing **metal leaching processes** from lithium-ion battery materials.

This repository contains a full pipeline for **forward modeling, reverse process prediction, and active learning** to better understand and optimize the extraction of critical metals such as **Lithium (Li), Cobalt (Co), Manganese (Mn), and Nickel (Ni)**. The workflow integrates data preprocessing, model training, uncertainty sampling, and interpretability analysis to support **AI-assisted process design for hydrometallurgical recycling**.

---

## Project Overview

The goal of this project is to apply modern machine learning methods to predict and optimize metal recovery from battery leaching experiments. The pipeline combines predictive modeling with experimental design techniques to better understand how process parameters influence extraction efficiency.

The workflow includes:

**Forward Modeling** – Predict metal extraction efficiencies from experimental conditions.

**Reverse Modeling** – Predict optimal process conditions from desired metal recovery targets.

**Active Learning** – Identify experimental conditions with high model uncertainty to guide future experiments.

**Model Explainability** – Interpret model behavior using SHAP feature importance and dependence analysis.

These tools help researchers explore relationships between feed composition, reagent type, temperature, reaction time, reagent concentration, and reducing agents and their influence on metal recovery efficiency.

---

## Features

### Machine Learning Models

The framework evaluates several regression algorithms, including:

* Random Forest
* Gradient Boosting
* Support Vector Regression
* Artificial Neural Networks
* k-Nearest Neighbors
* Decision Trees

The best performing models are automatically selected based on validation metrics.

---

### Forward Modeling

The forward model predicts metal extraction efficiencies from experimental parameters.

**Inputs**

* Feed composition
* Leaching reagent
* Reducing agent
* Temperature
* Time
* Reagent concentration

**Outputs**

* Lithium extraction (%)
* Cobalt extraction (%)
* Manganese extraction (%)
* Nickel extraction (%)

---

### Reverse Modeling

The reverse model predicts optimal process conditions for a desired extraction target.

Example:

Desired output:

* Li extraction = 95%
* Co extraction = 90%
* Mn extraction = 85%
* Ni extraction = 92%

Model prediction:

* Temperature
* Time
* Reagent type
* Reducing agent
* Reagent concentration
* Feed composition

This approach enables AI-assisted process design for battery recycling experiments.

---

### Active Learning

The pipeline includes a **Bayesian Active Learning (BAL)** strategy that helps identify the most informative experiments.

The process works as follows:

1. A Random Forest ensemble model is trained.
2. Prediction uncertainty is calculated across individual trees.
3. Samples with the highest uncertainty are identified.
4. These samples are suggested as new experiments.

This method helps researchers prioritize experiments that will most improve the model.

---

### Model Interpretability

The framework uses **SHAP (SHapley Additive Explanations)** to interpret model predictions.

Generated visualizations include:

* Feature importance plots
* SHAP summary plots
* SHAP dependence plots
* Feature interaction plots

These plots help reveal how experimental variables influence metal recovery.

---

## Installation

Clone the repository:

git clone https://github.com/mayafetzer/MetalLeaching.git
cd MetalLeaching

Create a virtual environment:

conda create -n metalleach python=3.10
conda activate metalleach

Install dependencies:

pip install -r requirements.txt

Required Python packages include:

* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib
* shap
* streamlit

---

## Running the Pipeline

The pipeline can be run using the main notebook:

notebooks/MetalLeachingPipeline.ipynb

or by running the Python training script:

python src/train_models.py

The pipeline performs the following steps:

1. Clean and preprocess the dataset
2. Train multiple machine learning models
3. Evaluate model performance
4. Generate explainability plots
5. Save trained models and results

## Applications

This framework can be applied to several areas of materials and chemical engineering, including:

* Lithium-ion battery recycling
* Hydrometallurgical metal extraction
* Process optimization
* Experimental design
* Materials discovery

---

## Future Work

Planned extensions include:

* Bayesian optimization for process design
* Deep learning surrogate models
* Multi-objective optimization
* Integration with laboratory automation
* Real-time experimental recommendation systems

---

## Citation

If you use this repository in academic work, please cite:

Machine Learning Assisted Optimization of Metal Leaching Processes


May A. Fetzer
Machine Learning for Sustainable Materials Processing


https://metalleaching.streamlit.app/
