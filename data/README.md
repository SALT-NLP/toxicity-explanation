# Datasets

## Social Bias Frames

To download social bias frames dataset, visit this [link](https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz), unzip the file
and place the train/dev/test files in this folder.

Here are some commands:

```
wget https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz
tar zxvf SBIC.v2.tgz
```

The code was written with the SBIC.v2.\*.csv files in mind. All credit for this dataset goes to the original authors:

Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A Smith & Yejin Choi (2020).
Social Bias Frames: Reasoning about Social and Power Implications of Language. ACL

## Implicit Hate Dataset

To download the implicit hate dataset, visit this [link](https://github.com/gt-salt/implicit-hate), navigate to the "Where can I download the data?"
section, take the survey to receive permission. Please note that the implicit hate dataset does not contain the same toxicity classifications as the Social Bias Frames dataset. Train a classifier on the toxicity classifications the implicit hate dataset does contain, or use a classifier trained on a different dataset.

To prepare the dataset, unzip the download and place the resulting directory (called "implicit-hate-corpus") in this data directory. Then call the script, like so:

```
python prep_implicit_hate_data.py
```

All credit for this dataset goes to the original authors:

ElSherief, M., Ziems, C., Muchlinski, D., Anupindi, V., Seybolt, J., De Choudhury, M., & Yang, D. (2021). Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).
