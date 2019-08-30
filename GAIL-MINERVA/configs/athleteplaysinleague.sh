#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/athleteplaysinleague/"
vocab_dir="datasets/data_preprocessed/athleteplaysinleague/vocab"
total_iterations=1000
path_length=3
hidden_size=50
embedding_size=50
batch_size=1
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/athleteplaysinleague/"
load_model=0
model_load_dir=""
nell_evaluation=1
