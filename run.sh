#!/usr/bin/env bash

set -eu

python train.py --out_fold ./models/ --gpu "3"
