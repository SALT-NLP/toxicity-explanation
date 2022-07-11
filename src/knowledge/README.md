# Implicit/Explicit Knowledge Models

## Explicit Knowledge Models

### Training the Explicit Knowledge Model
To train the explicit knowledge model, first download conceptnet data:

```
mkdir data
cd data
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
```

Then cd back out of the data directory and run the train file. An example command below trains the explicit knowledge model by querying for 5 additional 
knowledge tuples from ConceptNET, and directly concatenates them to the input (model_type flag should be set to "input").

```
python train.py --model_type input --data_file ../../data/SBIC.v2.trn.csv
```

To test the trained knowledge model, run the test.py file. Pass in the same model_type and k value as the trained model. An example command for a model 
trained per the settings in the command above is given below:

```
python test.py --model_type input --data_file ../../data/SBIC.v2.trn.csv --model model/bart_knowledge_k_25/
```



### Training the Implicit Knowledge Model
