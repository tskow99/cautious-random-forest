# Cautious-random-forest
## EN.675 Final Project 
The code found in this repository was developed in fufillment of EN.675 Final Project assignment, completed by @tskow99, @davfeldm17 and @jehuddleston. 

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
