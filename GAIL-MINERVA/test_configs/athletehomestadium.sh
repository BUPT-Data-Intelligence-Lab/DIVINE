#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/athletehomestadium/"
vocab_dir="datasets/data_preprocessed/athletehomestadium/vocab"
total_iterations=1000
path_length=3
hidden_size=50
embedding_size=50
batch_size=1
beta=0.05
Lambda=0.02
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/test_results/athletehomestadium"
load_model=1
model_load_dir="/media/dell_2/lrp/Multi-DIVINE/GAIL/output/athletehomestadium/test/model/model.ckpt"
nell_evaluation=1
