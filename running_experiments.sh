#!/usr/bin/env bash



dataset=$1
epochs=$2
n_class=$3
k=10

python GraphClassifier.py -n $dataset -c $n_class -k $k -e $epochs -f 3 -r
python GraphClassifier.py -n $dataset -c $n_class -k $k -e $epochs -f 3 -nr
python ConvNetPatchy.py -n $dataset -c $n_class -k $k -e $epochs -r
python ConvNetPatchy.py -n $dataset -c $n_class -k $k -e $epochs -nr


#python GraphClassifier.py -dataset_name 'DD' -w '284' -k 10 -r -exp 'no_relabel'
#python GraphClassifier.py -dataset_name 'DD' -w '284' -k 10 -exp 'with_relabel'


