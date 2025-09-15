 #!/usr/bin/env bash

# Specify the device for Hugging Face model inference; adjust as needed
HF_DEVICE="cuda:5"

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
    "adaptive-k-noclass-ignore5p"
    # "zero_shot"
    # "self-route"
    # 1000 
    # 5000 
    # 10000 
    # 25000 
    # 50000
    # "full_context"
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
LOG_FILE="${BASE_DIR}evaluation_log_${DATASET}.txt"
echo "Starting experiments at $(date)" > "$LOG_FILE"
EMBEDDING_TAG="${EMBEDDING_MODEL##*/}"

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
                PYTHON_ARGS="--zero_shot"

            elif [ "$RETRIEVAL_TOKENS" = "full_context" ]; then
                EXPERIMENT_TYPE="full-context"
                PYTHON_ARGS="--full_context"
            elif [ "$RETRIEVAL_TOKENS" = "adaptive-k-noclass-ignore5p" ]; then
                EXPERIMENT_TYPE="adaptive-k-noclass-ignore5p"
                PYTHON_ARGS="--adaptive_retrieval --retrieval_strategy largest_gap --ignore_extreme 0.05 --ignore_extreme_tail 0.1 --retrieve_more 5"
            elif [ "$RETRIEVAL_TOKENS" = "self-route" ]; then
                EXPERIMENT_TYPE="self-route"
                PYTHON_ARGS="--fixed_retrieval_tokens 5000"
            else
                RETRIEVAL_TOKENS_TAG="$(( RETRIEVAL_TOKENS / 1000 ))k"
                EXPERIMENT_TYPE="fixed-${RETRIEVAL_TOKENS_TAG}"
                PYTHON_ARGS="--fixed_retrieval_tokens ${RETRIEVAL_TOKENS}"
            fi

            
            if [ "$EXPERIMENT_TYPE" == "zeroshot" ]; then
                PROMPT_TEMPLATE="${BASE_DIR}templates/qa_zeroshot_template.txt"
            # if not zeroshot, thinking_budget is not 0, or qwen3 is included in MODEL_NAME, use template_reasoning_model
            elif [ $THINKING_BUDGET -ne 0 ] || [[ $MODEL_NAME == *"Qwen3"* ]] || [[ $MODEL_NAME == *"o4"* ]]; then
                PROMPT_TEMPLATE="${BASE_DIR}templates/holobench_template_reasoning_model.txt" 
            elif [ "$RETRIEVAL_TOKENS" = "self-route" ]; then
                EXPERIMENT_TYPE="self-route"
                PROMPT_TEMPLATE="${BASE_DIR}templates/holobench_template_self_route.txt"
            else
                PROMPT_TEMPLATE="${BASE_DIR}templates/holobench_template.txt"
            fi

            # Examples this handles:
            #  - "gpt-4o-2024-08-06"             → "gpt-4o"
            #  - "fireworks/.../qwen3-30b-a3b"    → "qwen3-30b-a3b"
            #  - "gemini/.../gemini-2.5-flash-preview-04-17"
            #                                     → "gemini-2.5-flash"
            #  - "gemini-2.0-flash-lite-001"      → "gemini-2.0-flash"
            #  - "gemini-2.0-flash-001"           → "gemini-2.0-flash"
            #  - "gemini-2.0-flash-thinking-exp-01-21"
            #                                     → "gemini-2.0-flash-thinking"


            # Construct the results directory path based on embedding, model, info, and retrieval tags
            RESULTS_DIR="${BASE_DIR}RAG_results/holobench/${EMBEDDING_TAG}/${MODEL_TAG}/info${INFO_TAG}/${EXPERIMENT_TYPE}"

            # Create results directory if it does not exist
            mkdir -p "${RESULTS_DIR}"

            # Define output file paths
            RETRIEVAL_RESULTS_FILE="${RESULTS_DIR}/retrieval_results.json"
            GENERATION_RESULTS_FILE="${RESULTS_DIR}/generation_results.json"

            # Run the main solver script with the specified parameters
            python "${BASE_DIR}solve.py" \
                --thinking_budget ${THINKING_BUDGET} \
                --task ${DATASET} \
                --num_samples 90 \
                --holobench_split all \
                --max_context_size 100000 \
                --info_amount "${INFO_AMOUNT}" \
                --merge_strategy uniform \
                --preprocessed_data "${BASE_DIR}dataset_src/holobench_${INFO_TAG}.dataset" \
                --retriever_model "${EMBEDDING_MODEL}" \
                --device cpu \
                --batch_size 16 \
                --return_true_k \
                ${PYTHON_ARGS} \
                --use_sentence_transformer \
                --use_faiss \
                --return_true_k \
                --generation_model "${MODEL_NAME}" \
                --generation_prompt_template ${PROMPT_TEMPLATE} \
                --classifier_prompt_template "${BASE_DIR}templates/classifier_hybrid_template.txt" \
                --classifier_examples "${BASE_DIR}templates/classifier_examples.json" \
                --max_generation_tokens 16384 \
                --retrieval_results_file "${RETRIEVAL_RESULTS_FILE}" \
                --generation_results_file "${GENERATION_RESULTS_FILE}" \
                --hf_generation_device "${HF_DEVICE}" \
                --overwrite_results 
                # --run_async

            # Log completion
            echo "Completed evaluation: Dataset=$DATASET, Model=$MODEL_NAME, Experiment Type=$EXPERIMENT_TYPE" | tee -a "$LOG_FILE"
        done
    done
done

echo "All evaluations completed at $(date)" | tee -a "$LOG_FILE"