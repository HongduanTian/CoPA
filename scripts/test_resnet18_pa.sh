#! /bin/bash
ulimit -n 50000
export META_DATASET_ROOT=./meta_dataset
export RECORDS= # Path/to/data
CUDA_VISIBLE_DEVICES=<gpu_id> python test_extractor_pa.py --model.name=url --model.dir ./url \
                            --test.type=standard \
                            --seed=42 \
                            --experiment.name=seed42