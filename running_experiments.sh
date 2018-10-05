#!/usr/bin/env bash


python GraphClassifier.py -dataset_name 'MUTAG' -w '18' -k 10 -r -exp 'no_relabel'
python GraphClassifier.py -dataset_name 'MUTAG' -w '18' -k 10 -exp 'with_relabel'

python GraphClassifier.py -dataset_name 'DD' -w '284' -k 10 -r -exp 'no_relabel'
python GraphClassifier.py -dataset_name 'DD' -w '284' -k 10 -exp 'with_relabel'


