# Toxicity Explanation Models

This repo contains code for the following paper

```Sridhar, Rohit, and Diyi Yang. "Explaining Toxic Text via Knowledge Enhanced Text Generation." NAACL 2022. ```

## Getting Started

These instructions will get you started with running MixGEN and other knowledge informed models described in the paper.

### Requirements
* Python 3.7 or higher
* Pytorch (>= 1.8.1)
* Huggingface transformers (4.5.1)

### Code Structure
```
|__ baselines
  |__ README.md
  |__ train.py
  |__ test.py
  |__ gpt_utils.py --> utils for baselines (all gpt models)
  |__ grid_search.sh --> code to run grid search to tune hyperparameters
|__ data --> This folder should contain all the main datasets that are trained/tested on
  |__ README.md --> Instructions to download data
|__ shared
  |__ README.md
  |__ utils.py --> Main utils used by most modules in this repo
  |__ significance_test.py --> Code to run significance tests
  |__ multi_view_trainer.py --> Code to override huggingface trainer and pytorch sampler classes
|__ src
  |__ seq2seq --> Code to train/test Expert Knowledge models
    |__ README.md
    |__ train.py
    |__ test.py
    |__ seq2seq.py --> Class definition for BART with join embedding
    |__ seq2seq_utils.py
    |__ grid_search.sh
    |__ classification --> Contains code to train BERT classifiers used for join embedding
      |__ README.md
      |__ train_classifier.ipynb
      |__ test_classifier.ipynb
  |__ knowledge --> Code to train/test Explicit/Implicit Knowledge models
    |__ README.md --> Instructions to train/test Explicit/Implicit Knowledge models
    |__ train.py --> For training Explicit Knowledge models
    |__ test.py --> For testing Explicit Knowledge models
    |__ train_imp.py --> For training Implicit Knowledge models
    |__ test_imp.py --> For testing Implicit Knowledge models
    |__ knowledge.py --> Class definition for BART with Explicit Knowledge
    |__ knowledge_utils.py
    |__ grid_search.sh --> Grid Search on Explicit Knowledge models
    |__ grid_search_imp.sh --> Grid Search on Implicit Knowledge models
  |__ ensemble --> Code to train/test MixGEN models
    |__ README.md --> Instructions to train/test MixGEN models
    |__ train.py
    |__ test.py
    |__ ensemble.py --> Class definition for BART based MixGEN MultiView
    |__ ensemble_utils.py --> Main utils for MixGEN
    |__ generation_utils.py --> Text generator class for MixGEN MultiView
```

### Setup
This tutorial assumes the use of conda, but any other means by which to control the code environment should work. First clone the repository, and cd into the directory, like so:

```
git clone https://github.com/GT-SALT/toxicity-explanation
cd toxicity-explanation
```

Next, create a conda environment and install the requirements

```
conda create -n tox_env python=3.7
conda activate tox_env
pip install -r requirements.txt
```

You can replace tox_env with a custom name! Setup should now be complete.

### Downloading the Data
To download the data, please see the README in the data/ directory.

### Training and Testing Models
The baseline models can be trained/tested in the baselines/ directory. The Expert knowledge models can be trained/tested in the src/seq2seq/ directory. 
The Explicit/Implicit knowledge models can be trained/tested in the src/knowledge/ directory, while the MixGEN models can be trained in the src/ensemble/ directory. The READMEs in the individual directories contain instructions on training and testing.
