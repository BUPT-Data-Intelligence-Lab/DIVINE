# GAIL-MINERVA
MINERVA ([Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851)) enhanced by our GAIL-based framework DIVINE

The implementation of GAIL-MINERVA is based on the code released by https://github.com/shehzaadzd/MINERVA

## Requirements
To install the various python dependences (including tensorflow)
```
pip install -r requirements.txt
```

## Dataset

The preprocessed NELL-995  dataset can be downloaded [here](<https://drive.google.com/open?id=1SXK42ImtCkdTE-gxvfhJ_27K21vhqYKy>) and should be located in the [datasets](<https://github.com/Ruiping-Li/DIVINE/tree/master/GAIL-MINERVA/datasets>) directory.

## Training

The hyperparameter configs for each experiments are in the [configs](<https://github.com/Ruiping-Li/DIVINE/tree/master/GAIL-MINERVA/configs>) directory. 
To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the config file. For example, 
```
sh run.sh configs/athelehomestadium.sh
```
To start the whole tranining experiment, just do
```
python train_nell.py
```

## Testing
We are also releasing pre-trained GAIL-MINERVA models for testing. They can be downloaded [here](<https://drive.google.com/open?id=1KknUAhI9aVvgi08K0IZxrGpmSveBdU3N>) and should be located in the [output](<https://github.com/Ruiping-Li/DIVINE/tree/master/GAIL-MINERVA/output>) directory. To load the model, set the ```load_model``` to 1 in the config file (default value 0) and ```model_load_dir``` to point to the saved_model.
To start a particular experiment, just do
```
sh run.sh test_configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the config file. For example, 
```
sh run.sh test_configs/athelehomestadium.sh
```
To start the whole testing  experiment, just do
```
python test_nell.py
```

## Output
The metrics used for evaluation are Hits@{1,3}, MAP and MRR. Along with this, the code also outputs the answers GAIL-MINERVA reached in a file.


## Citation
If you use this code, please cite our paper
