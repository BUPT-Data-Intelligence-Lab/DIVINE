#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/nell-995/"
vocab_dir="datasets/data_preprocessed/nell-995/vocab"
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
base_output_dir="output/test_results/nell-995"
load_model=1
model_load_dir="/media/dell_2/lrp/Multi-DIVINE/GAIL/output/nell-995/test/model/model.ckpt"
nell_evaluation=1
