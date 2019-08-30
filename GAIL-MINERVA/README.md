# GAIL-MINERVA
MINERVA ([Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851)) enhanced by our GAIL-based framework DIVINE

The implementation of GAIL-MINERVA is based on the code released by https://github.com/shehzaadzd/MINERVA

## Requirements
To install the various python dependences (including tensorflow)
```
pip install -r requirements.txt
```

## Training
The hyperparam configs for each experiments are in the [configs](https://github.com/shehzaadzd/MINERVA/tree/master/configs) directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the config file. For example, 
```
sh run.sh configs/athelehomestadium.sh
```

## Testing
We are also releasing pre-trained GAIL-MINERVA models for testing. They are located in the  [saved_models](https://github.com/shehzaadzd/MINERVA/tree/master/saved_models) directory. To load the model, set the ```load_model``` to 1 in the config file (default value 0) and ```model_load_dir``` to point to the saved_model.

## Output
The code outputs the evaluation of MINERVA on the datasets provided. The metrics used for evaluation are Hits@{1,3,5,10,20} and MRR (which in the case of Countries is AUC-PR). Along with this, the code also outputs the answers MINERVA reached in a file.


## Citation
If you use this code, please cite our paper
