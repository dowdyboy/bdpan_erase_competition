#!/usr/bin/env bash

# module add nvhpc/nvhpc/21.7
# module add cuda/11.6
# source ~/my_envs/envs/bin/activate
# module add nvhpc/nvhpc/21.7
# module add cuda/11.6
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/software/cuda/11.6/lib64/

PYTHONPATH="./":$PYTHONPATH \
python ./train.py $*

