#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

RESULTS_DIR=$1
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
TASK_NAME="json"

# Experiments across methods

python -m experiments \
    base-lm \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --result-dir $RESULTS_DIR


python -m experiments \
    lcd \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --result-dir $RESULTS_DIR


python -m experiments \
    sample-rerank \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 10 \
    --result-dir $RESULTS_DIR


python -m experiments \
    twisted-smc \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 10 \
    --ess-threshold 0.90 \
    --result-dir $RESULTS_DIR


python -m experiments \
    awrs-smc \
    --model-name $MODEL_NAME \
    --task $TASK_NAME \
    --use-chat-format \
    --num-particles 5 \
    --ess-threshold 0.5 \
    --result-dir $RESULTS_DIR


# Varying number of particles for AWRS SMC and Twisted SMC

for num_particles in 1 2 5 10; do
    python -m experiments \
        awrs-smc \
        --model-name $MODEL_NAME \
        --task $TASK_NAME \
        --use-chat-format \
        --num-particles $num_particles \
        --ess-threshold 0.5 \
        --result-dir $RESULTS_DIR
done


for num_particles in 10 20 40; do
    python -m experiments \
        twisted-smc \
        --model-name $MODEL_NAME \
        --task $TASK_NAME \
        --use-chat-format \
        --num-particles $num_particles \
        --ess-threshold $(echo "scale=3; ($num_particles - 1)/$num_particles" | bc) \
        --result-dir $RESULTS_DIR
done
