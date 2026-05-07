#!/bin/bash

THRESHOLD=0.001
ACDC_REPO=/scratch/network/mc3803/Automatic-Circuit-Discovery
EDGE_REPO=/scratch/network/mc3803/Edge-Pruning
DATASET=$EDGE_REPO/data/datasets/gt
JSON_OUT=$EDGE_REPO/data/acdc_results/gt-t${THRESHOLD}-graph.json
CKPT_OUT=$EDGE_REPO/data/acdc_checkpoints/gt-t${THRESHOLD}

mkdir -p $EDGE_REPO/data/acdc_results
mkdir -p $CKPT_OUT
mkdir -p $EDGE_REPO/joblog

sbatch --partition=gpu --gres=gpu:1 --mem=32G --time=12:00:00 --constraint=gpu80 \
    --job-name=acdc_gt \
    --output=$EDGE_REPO/joblog/acdc_gt_t${THRESHOLD}_%j.out \
    --wrap="cd $EDGE_REPO && \
    module load anaconda3/2025.12 && \
    conda activate /scratch/network/mc3803/envs/acdc2 && \
    export TRANSFORMERS_OFFLINE=1 && \
    export HF_HOME=/scratch/network/mc3803/.cache/huggingface && \
    echo '[Step 1] Running ACDC on GT...' && \
    python $ACDC_REPO/acdc_gt_eval.py \
        --threshold $THRESHOLD \
        --dataset-path $DATASET \
        --max-train-examples 150 \
        --max-num-epochs 100000 \
        --out-json-path $JSON_OUT \
        --out-pickle-path-final $EDGE_REPO/data/acdc_results/gt-t${THRESHOLD}-graph.pkl && \
    echo '[Step 2] Converting to checkpoint...' && \
    python $EDGE_REPO/acdc_to_checkpoint.py \
        --acdc-json-path $JSON_OUT \
        --out-dir $CKPT_OUT && \
    echo '[Step 3] Evaluating circuit...' && \
    python $EDGE_REPO/src/eval/eval_acdc_gt.py \
        --acdc-json-path $JSON_OUT \
        --data-path $DATASET"