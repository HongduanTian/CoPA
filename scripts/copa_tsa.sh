#! /bin/bash
ulimit -n 50000
export META_DATASET_ROOT=./meta_dataset
export RECORDS= #Path/to/data
CUDA_VISIBLE_DEVICES=<gpu_id> python copa_tsa.py --model.name=url \
                            --model.dir ./url \
                            --test.type=standard \
                            --encoder.type=linear \
                            --SCE.tau=2.0 \
                            --seed=42 \
                            --exp_dir_name=copa_tsa_all \
                            --experiment.name=seed42