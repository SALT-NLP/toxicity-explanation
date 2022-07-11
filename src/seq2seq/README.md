# Expert Knowledge Models

### Training an Expert Knowledge Model

To train the expert knowledge models, first train a classification model on a toxicty classification. You can do so by following the Colab notebooks in 
the classification/ subdirectory.

To train an Expert knowledge informed model run the train.py file and pass in the toxicity classification model path. An example command is given below. 
The model will be trained using the toxicity classifier for the harm intent classification in the Social Bias Frames dataset

```
python train.py --join --classifiers classification/model/intentYN/ --data_file ../../data/SBIC.v2.trn.csv
```

Multiple classifiers may be passed into the classifiers argument to train using multiple knowledge sources.

### Testing an Expert Knowledge model

To test an expert knowledge model, pass in the path to the trained model and a path to the classifier used to feed expert knowledge to the model. An 
example command testing a model trained using the harm intent classifier is given below:

```
python test.py --model model/bart_join_grp_checkpoint-3epoch/ --classifiers classification/model/intentYN/checkpoint-898/ --data_file ../../data/SBIC.v2.dev.csv
```
