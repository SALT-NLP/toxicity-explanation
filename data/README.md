# Datasets

## Social Bias Frames

To download social bias frames dataset, visit this [link](https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz), unzip the file
and place the train/dev/test files in this folder.

Here are some commands:

```
wget https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz
tar zxvf SBIC.v2.tgz
```

The code was written with the SBIC.v2.*.csv files in mind.

## Implicit Hate Dataset

To download the implicit hate dataset, visit this [link](https://github.com/gt-salt/implicit-hate), navigate to the "Where can I download the data?"
section, take the survey to receive permission and add the train/test files to this directory. Please note that the implicit hate dataset does not
contain toxicity classifications and thus cannot use the Expert Knowledge model's join embedding classifier directly. Instead, train a classifier on
a dataset that does contain toxicity classifications and use it.
