# Cautious-random-forest
## EN.675 Final Project 

This repository explores methods for implementing cautiousness in Random Forest classifiers for binary classification tasks. We investigate various algorithms that account for uncertainty during decision-making and evaluate their effectiveness across multiple datasets. A summary of our findings and analysis is provided in the accompanying paper available in this repository.


This project was developed in fufillment of EN.675 Final Project assignment, completed by @tskow99, @davfeldm17 and @jehuddleston. 

# Getting started
Clone the repository

`git clone git@github.com:tskow99/cautious-random-forest.git`

Install dependencies

`conda create --name crf python=3.9`

`conda activate crf`

`pip install -r requirements.txt`

## Code Layout

To validate the results from our report, run `results.ipynb`. The notebook will fit each of our classifiers to each of our datasets and run evaluation on all classifiers. 

Our classifier definitions are found in `classifiers.py`. `eval.py` contains the evaluation metrics specific to each classifier and `data.py` contains the data loaders for each dataset. 

## Data

We use the following datasets:

- Compas data from https://github.com/propublica
- Heart Disease dataset from https://archive.ics.uci.edu/dataset/45/heart+disease
- Breast Cancer Wisconsin (Diagnostic) from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Statlog (German Credit Data) from https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## Sources

Our methods and code is adapated from the following previouse work: 

- Haifei Zhang, Benjamin Quost, Marie-Hélène Masson,
Cautious weighted random forests,
Expert Systems with Applications,
Volume 213, Part A,
2023,
118883,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2022.118883.
(https://www.sciencedirect.com/science/article/pii/S0957417422019017)
Keywords: Cautious classification; Imprecise classification; Imprecise dirichlet model; Belief functions

- Ferri, C., Hernández-Orallo, J., & Modroiu, R. (2004). Cautious Classifiers. Workshop on ROC Analysis in Artificial Intelligence (ROCAI 2004), 27-36. https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf

