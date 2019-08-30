# GAIL-DeepPath
DeepPath ([DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning](https://arxiv.org/abs/1707.06690)) enhanced by our GAIL-based framework DIVINE

The implementation of GAIL-MINERVA is based on the code released by https://github.com/xwhan/DeepPath

## Requirements
To install the various python dependences (including tensorflow)
```
pip install -r requirements.txt
```

## Dataset

The preprocessed NELL-995  dataset can be downloaded [here](<https://drive.google.com/open?id=1NcE3eQ3MRcchvP_SoIE-J8XiePfiLVgj>) and should be located in the [NELL-995](<https://github.com/Ruiping-Li/DIVINE/tree/master/GAIL-DeepPath/NELL-995>) directory.

## Training

The training scripts are in the [scripts](<https://github.com/Ruiping-Li/DIVINE/tree/master/GAIL-DeepPath/scripts>) directory. 
To start the whole tranining experiment, just do
```
cd scripts/
python all_demo_sampling.py
python all_pretrain.py
python all_retrain.py
```

## Testing
We are also releasing pre-trained GAIL-DeepPath models for testing. They can be downloaded [here](<https://drive.google.com/open?id=1KknUAhI9aVvgi08K0IZxrGpmSveBdU3N>) and should be located in the [models](<https://drive.google.com/open?id=1L0YV5vJ3GkiTcjVhi0tIYd2Bat-CMfd6>) directory. 
To start the whole testing  experiment, just do
```
cd scripts/
python all_link_prediction.py
python all_fact_prediction.py
```

## Output
The metrics used for evaluation are Hits@{1,3}, MAP and MRR. Along with this, the code also outputs the answers GAIL-DeepPath reached in a file.


## Citation
If you use this code, please cite our paper
