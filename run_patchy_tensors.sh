#!/usr/bin/env bash


dataset=$1

python create_patchy_tensor.py -n $dataset
python create_patchy_tensor.py -n $dataset



#python GraphClassifier.py -dataset_name 'DD' -w '284' -k 10 -r -exp 'no_relabel'
#python GraphClassifier.py -dataset_name 'DD' -w '284' -k 10 -exp 'with_relabel'


