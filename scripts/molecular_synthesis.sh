#!/bin/bash

if [ -z "$1" ]; then
    RESULTS_DIR="results/molecular_synthesis"
else
    RESULTS_DIR=$1
fi

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
TASK_NAME="molecular-synthesis"

# There is a lot of variance in these results, so we run 10 replicates.

# Experiments across methods

python -m experiments \
    base-lm \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --result-dir $RESULTS_DIR \
    --n-replicates 10


python -m experiments \
    lcd \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --result-dir $RESULTS_DIR \
    --n-replicates 10


python -m experiments \
    sample-rerank \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --num-particles 10 \
    --result-dir $RESULTS_DIR \
    --n-replicates 10


python -m experiments \
    twisted-smc \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --num-particles 10 \
    --ess-threshold 0.90 \
    --result-dir $RESULTS_DIR \
    --n-replicates 10

python -m experiments \
    awrs-smc \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 5 \
    --ess-threshold 0.5 \
    --result-dir $RESULTS_DIR \
    --n-replicates 10
