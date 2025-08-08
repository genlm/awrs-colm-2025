#!/bin/bash

if [ -z "$1" ]; then
    RESULTS_DIR="results/pattern_matching"
else
    RESULTS_DIR=$1
fi

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
TASK_NAME="pattern-matching"
N_REPLICATES=2

# Experiments across methods

python -m experiments \
    base-lm \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --result-dir $RESULTS_DIR \
    --n-replicates $N_REPLICATES


python -m experiments \
    lcd \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --result-dir $RESULTS_DIR \
    --n-replicates $N_REPLICATES


python -m experiments \
    sample-rerank \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 10 \
    --result-dir $RESULTS_DIR \
    --n-replicates $N_REPLICATES


python -m experiments \
    twisted-smc \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 10 \
    --ess-threshold 0.90 \
    --result-dir $RESULTS_DIR \
    --n-replicates $N_REPLICATES


python -m experiments \
    awrs-smc \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 5 \
    --ess-threshold 0.5 \
    --result-dir $RESULTS_DIR \
    --n-replicates $N_REPLICATES
