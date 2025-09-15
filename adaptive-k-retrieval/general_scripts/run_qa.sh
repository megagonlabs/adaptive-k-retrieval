#!/usr/bin/env bash

# Specify the device for Hugging Face model inference; adjust as needed
HF_DEVICE="cuda:4"

# Base directory for this project; adjust as needed
BASE_DIR="./"

# Define arrays of parameters
DATASETS=(
    "nq" 
    "triviaqa" 
    "hotpotqa"
    )
MODEL_NAMES=(
    # "vertex_ai/gemini-2.5-flash-preview-04-17"
    # "fireworks_ai/accounts/fireworks/models/llama4-scout-instruct-basic"
    # "fireworks_ai/accounts/fireworks/models/llama4-maverick-instruct-basic"
    "gpt-4o-2024-08-06"
    # "gpt-4o-mini-2024-07-18"
)
THINKING_BUDGET=0
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
# EMBEDDING_MODEL="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
RETRIEVAL_TOKENS_LIST=(
    "adaptive-k-noclass"
    # "self-route"
    # "zero_shot"
    # "full_context"
    # 1000 
    # 5000 
    # 10000 
    # 25000 
    # 50000
    )

# Extract the last segment of the embedding model path
EMBEDDING_TAG="${EMBEDDING_MODEL##*/}"

# Create a log file to track progress
LOG_FILE="${BASE_DIR}experiment_log.txt"
echo "Starting experiments at $(date)" > "$LOG_FILE"

# Loop through all combinations
for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        for RETRIEVAL_TOKENS in "${RETRIEVAL_TOKENS_LIST[@]}"; do
            # Process model name to create tag
            BASE="${MODEL_NAME##*/}"             # drop any path prefix
            STEP1="${BASE%-preview*}"            # remove "-preview…" suffix
            STEP2="${STEP1%-????-??-??}"         # remove trailing "-YYYY-MM-DD"
            STEP3="${STEP2%-exp*}"               # remove "-exp…" and beyond
            STEP4="${STEP3%-lite}"               # remove "-lite" suffix
            MODEL_TAG="${STEP4%-001}"            # remove trailing "-001"

            # if thinking_budget is not 0, add it to the model tag
            if [ $THINKING_BUDGET -ne 0 ]; then
                MODEL_TAG="${MODEL_TAG}-${THINKING_BUDGET}"
            fi

            # Determine the experiment type and construct results directory path
            if [ "$RETRIEVAL_TOKENS" = "zero_shot" ]; then
                EXPERIMENT_TYPE="zeroshot"
                PYTHON_ARGS="--zero_shot"
                PROMPT_TEMPLATE="${BASE_DIR}templates/qa_zeroshot_template.txt"
            elif [ "$RETRIEVAL_TOKENS" = "full_context" ]; then
                EXPERIMENT_TYPE="full-context"
                PYTHON_ARGS="--full_context"
                PROMPT_TEMPLATE="${BASE_DIR}templates/qa_template.txt"
            elif [ "$RETRIEVAL_TOKENS" = "adaptive-k-noclass" ]; then
                EXPERIMENT_TYPE="adaptive-k-noclass"
                PYTHON_ARGS="--adaptive_retrieval --retrieval_strategy largest_gap --ignore_extreme_tail 0.1 --retrieve_more 5"
                PROMPT_TEMPLATE="${BASE_DIR}templates/qa_template.txt"
            elif [ "$RETRIEVAL_TOKENS" = "self-route" ]; then
                EXPERIMENT_TYPE="self-route"
                PYTHON_ARGS="--fixed_retrieval_tokens 5000"
                PROMPT_TEMPLATE="${BASE_DIR}templates/qa_self_route_template.txt"
            else
                RETRIEVAL_TOKENS_TAG="$(( RETRIEVAL_TOKENS / 1000 ))k"
                EXPERIMENT_TYPE="fixed-${RETRIEVAL_TOKENS_TAG}"
                PYTHON_ARGS="--fixed_retrieval_tokens ${RETRIEVAL_TOKENS}"
                PROMPT_TEMPLATE="${BASE_DIR}templates/qa_template.txt"
            fi

            # Construct the results directory path
            RESULTS_DIR="${BASE_DIR}RAG_results/${DATASET}/${EMBEDDING_TAG}/${MODEL_TAG}/${EXPERIMENT_TYPE}"

            # Create results directory if it does not exist
            mkdir -p "${RESULTS_DIR}"

            if [ ! -d "$RESULTS_DIR" ]; then
                echo "Error: Directory $RESULTS_DIR does not exist."
                exit 1
            fi

            RETRIEVAL_RESULTS_FILE=${RESULTS_DIR}/retrieval_results.json
            GENERATION_RESULTS_FILE=${RESULTS_DIR}/generation_results.json

            # Log the current experiment
            echo "Running experiment: Dataset=$DATASET, Model=$MODEL_NAME, Experiment Type=$EXPERIMENT_TYPE" | tee -a "$LOG_FILE"

            # Run the experiment
            python "${BASE_DIR}solve.py" \
                --thinking_budget ${THINKING_BUDGET} \
                --task ${DATASET} \
                --num_samples 100 \
                --drop_duplicates \
                --device cpu \
                --retriever_model "${EMBEDDING_MODEL}" \
                --return_true_k \
                ${PYTHON_ARGS} \
                --use_sentence_transformer \
                --use_faiss \
                --generation_model ${MODEL_NAME} \
                --generation_prompt_template ${PROMPT_TEMPLATE} \
                --classifier_prompt_template "${BASE_DIR}templates/classifier_hybrid_template.txt" \
                --classifier_examples "${BASE_DIR}templates/classifier_examples.json" \
                --max_generation_tokens 16384 \
                --retrieval_results_file ${RETRIEVAL_RESULTS_FILE} \
                --generation_results_file ${GENERATION_RESULTS_FILE} \
                --hf_generation_device "${HF_DEVICE}" \
                --overwrite_results
                # --run_async

            # Log completion
            echo "Completed experiment: Dataset=$DATASET, Model=$MODEL_NAME, Experiment Type=$EXPERIMENT_TYPE" | tee -a "$LOG_FILE"
        done
    done
done

echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"