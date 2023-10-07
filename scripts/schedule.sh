#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python src/train.py experiment=convnext-tiny-rubert-tiny
