 #!/usr/bin/env bash

# Base directory for this project; adjust as needed
BASE_DIR="./"

# Define arrays of parameters
DATASET="holobench"

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
INFO_AMOUNT_LIST=(
    5000
    10000
    25000
    50000
)

# Extract the last segment of the embedding model path
EMBEDDING_TAG="${EMBEDDING_MODEL##*/}"

# Create a log file to track progress
LOG_FILE="${BASE_DIR}evaluation_log.txt"
echo "Starting evaluations at $(date)" > "$LOG_FILE"

# Loop through all combinations
for INFO_AMOUNT in "${INFO_AMOUNT_LIST[@]}"; do
    INFO_TAG="$(( INFO_AMOUNT / 1000 ))k"
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
            elif [ "$RETRIEVAL_TOKENS" = "full_context" ]; then
                EXPERIMENT_TYPE="full-context"
            elif [ "$RETRIEVAL_TOKENS" = "self-route" ]; then
                EXPERIMENT_TYPE="self-route"
            elif [ "$RETRIEVAL_TOKENS" = "adaptive-k-noclass-ignore5p" ]; then
                EXPERIMENT_TYPE="adaptive-k-noclass-ignore5p"
            else
                RETRIEVAL_TOKENS_TAG="$(( RETRIEVAL_TOKENS / 1000 ))k"
                EXPERIMENT_TYPE="fixed-${RETRIEVAL_TOKENS_TAG}"
            fi

            if [ "$RETRIEVAL_TOKENS" = "self-route" ]; then
                PYTHON_ARGS="--self_route"
            else
                PYTHON_ARGS=""
            fi

            # Construct the results directory path
            RESULTS_DIR="${BASE_DIR}RAG_results/holobench/${EMBEDDING_TAG}/${MODEL_TAG}/info${INFO_TAG}/${EXPERIMENT_TYPE}"

            if [ ! -d "$RESULTS_DIR" ]; then
                echo "Warning: Directory $RESULTS_DIR does not exist. Skipping evaluation."
                continue
            fi

            RETRIEVAL_RESULTS_FILE=${RESULTS_DIR}/retrieval_results.json
            GENERATION_RESULTS_FILE=${RESULTS_DIR}/generation_results.json

            if [ ! -f "$RETRIEVAL_RESULTS_FILE" ] || [ ! -f "$GENERATION_RESULTS_FILE" ]; then
                echo "Warning: Results files not found in $RESULTS_DIR. Skipping evaluation."
                continue
            fi

            RETRIEVAL_EVAL_FILE=${RESULTS_DIR}/retrieval_eval.json
            GENERATION_EVAL_FILE=${RESULTS_DIR}/generation_eval.json

            # Log the current evaluation
            echo "Running evaluation: Dataset=$DATASET, Model=$MODEL_NAME, Experiment Type=$EXPERIMENT_TYPE" | tee -a "$LOG_FILE"

            # Run the evaluation
            python "${BASE_DIR}eval.py" \
                --task ${DATASET} \
                --retrieval_result_file ${RETRIEVAL_RESULTS_FILE} \
                --generation_result_file ${GENERATION_RESULTS_FILE} \
                --retrieval_eval_output_file ${RETRIEVAL_EVAL_FILE} \
                --generation_eval_score_file ${GENERATION_EVAL_FILE} \
                --eval_prompt_file "${BASE_DIR}templates/${DATASET}_eval_template.txt" \
                --post_process_prompt_file "${BASE_DIR}templates/${DATASET}_postproc_template.txt" \
                ${PYTHON_ARGS} 

            # Log completion
            echo "Completed evaluation: Dataset=$DATASET, Model=$MODEL_NAME, Experiment Type=$EXPERIMENT_TYPE" | tee -a "$LOG_FILE"
        done
    done
done

echo "All evaluations completed at $(date)" | tee -a "$LOG_FILE"