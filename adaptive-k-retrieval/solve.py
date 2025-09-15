"""Main program for running the pipeline."""
from datasets import Dataset
from tqdm import tqdm
from dotenv import load_dotenv
import torch
import json
import os
import random
from transformers import PreTrainedModel, PreTrainedTokenizer
from litellm import completion, acompletion
import asyncio  # for HoloBench experiments with async run

# Annotations, logging
import argparse
from typing import List, Dict, Optional, Any
import time
import logging
import warnings

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
# logging.basicConfig(level=logging.WARNING)

# Local
import classifier
from retriever import Retriever
from task_loader import TaskLoader
import utils

from dataclasses import dataclass


# set visible GPU - This will be controlled by device_map in HF loading now
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------- Fireworks‑like return‑value helpers ----------
@dataclass
class Usage:
    """Token‑count information identical to Fireworks .usage"""
    prompt_tokens: int
    completion_tokens: int

@dataclass
class ChoiceMessage:
    """Holds the assistant text"""
    content: str

@dataclass
class Choice:
    """Mimics .choices[0] structure"""
    message: ChoiceMessage

@dataclass
class HFResponse:
    """Top-level object returned by chat_request_hf"""
    choices: List[Choice]
    usage: Usage
# --------------------------------------------------------


def chat_request_hf(
    model: PreTrainedModel, # Modified to accept model object
    tokenizer: PreTrainedTokenizer, # Modified to accept tokenizer object
    prompt: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: int = 16384, # Changed default value to a more realistic one
    response_format_type: Optional[Dict[str, Any]] = None,
    ) -> HFResponse:
    """
    Hugging Face equivalent of the Fireworks chat_request helper,
    using pre-loaded model and tokenizer.

    Parameters
    ----------
    model : PreTrainedModel
        Pre-loaded Hugging Face model object.
    tokenizer : PreTrainedTokenizer
        Pre-loaded Hugging Face tokenizer object.
    prompt : str
        User message (single‑turn demo).
    temperature : float, optional
        Sampling temperature (0.0 = deterministic).
    top_p : float, optional
        Nucleus sampling probability mass.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate.
    response_format_type : dict, optional
        Not implemented; included only to keep the same signature.
    """
    if response_format_type is not None:
        warnings.warn(
            "response_format_type is currently unused and has no effect. "
            "It is included only to keep the same signature."
        )
    # 1. Model and tokenizer are received as arguments, so loading process is not included
    
    # 2. Build a chat‑style prompt (adjust markers to your model)
    # Note: Chat format may differ depending on the model
    # Qwen1.5/Qwen2 format:
    messages = [
        {"role": "user", "content": prompt}
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # --- Example for formats other than Qwen1.5/Qwen2 ---
    # chat_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n" # Original format

    # Ensure inputs are on the same device as the model
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[-1]

    # 3. Generation arguments mirroring Fireworks defaults (max_new_tokens is specified by argument)
    do_sample = temperature > 0.0 if temperature is not None else False
    gen_kwargs = {
        "max_new_tokens": max_new_tokens, # Use argument
        "temperature": temperature if do_sample else 0.0,
        "top_p": top_p if do_sample else None,
        "top_k": None,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id, # Usually same as eos_token_id for generation
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Get only the generated tokens, excluding the input part
    # Shape of output_ids[0] is (sequence_length,)
    generated_ids = output_ids[0][prompt_tokens:]
    completion_tokens = generated_ids.shape[-1]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 4. Wrap results to mimic Fireworks Response object
    return HFResponse(
        choices=[Choice(message=ChoiceMessage(content=generated_text))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


# Load the config
with open("config.json", "r") as f:
    config = json.load(f)


## GET CREDENTIALS
file_path = config["vertexai_credential_path"]

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

# Load other environment variables from .env file; these include OpenAI API key.
load_dotenv()


# Constants
SUPPORTED_TASKS = {"hotpotqa", "holobench", "nq", "triviaqa"}

HELMET_DIR = config["helmet_dir"]

QA_TASKS = ["nq", "triviaqa", "hotpotqa"]

QA_DEMO_FILE_NAMES = ["HELMET/data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
                      "HELMET/data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
                      "HELMET/data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl"]

QA_DEMO_FILES = [os.path.join(HELMET_DIR, file) for file in QA_DEMO_FILE_NAMES]

QA_DEV_FILE_NAMES = ["HELMET/data/kilt/nq-dev-multikilt_1000_k1000_dep6.jsonl",
                     "HELMET/data/kilt/triviaqa-dev-multikilt_1000_k1000_dep6.jsonl",
                     "HELMET/data/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl"]

QA_DEV_FILES = [os.path.join(HELMET_DIR, file) for file in QA_DEV_FILE_NAMES]

TASK_TO_DEMO_FILE = dict(zip(QA_TASKS, QA_DEMO_FILES))

TASK_TO_DEV_FILE = dict(zip(QA_TASKS, QA_DEV_FILES))

HOLOBENCH_SPLITS = {"wine_1", "college_2",
                    "flight_4", "store_1", "soccer_1", "all"}

HOLOBENCH_TASKS = {
    "holobench_5k",
    "holobench_10k",
    "holobench_25k",
    "holobench_50k",
}

MERGE_STRATEGIES = {"uniform", "begin", "end", "middle", "bimodal"}

RETRIEVAL_CATEGORIES = {"factoid", "aggregation"}

MAX_GENERATION_TOKENS = 16384


def load_data(task: str,
              num_samples: int,
              drop_duplicates: bool = True,
              debug: bool = False,
              **holobench_args) -> Dataset:
    """Load data for the specified task.
    
    Args:
        task (str): Task name. One of SUPPORTED_TASKS.
        num_samples (int): Number of samples to load.
        drop_duplicates (bool): Whether to drop duplicate samples.
        debug (bool): Whether to load the demo set (True) or the dev set (False).
        **holobench_args: Additional arguments for Holobench task loading.
        
    Returns:
        Dataset: Loaded dataset.
    """
    task_loader = TaskLoader(debug=debug)

    if task == "holobench":
        data = task_loader.load_holobench(
            split=holobench_args["holobench_split"],
            max_context_size=holobench_args["max_context_size"],
            info_amount=holobench_args["info_amount"],
            merge_strategy=holobench_args["merge_strategy"],
            num_samples=num_samples,
            preprocessed_data=holobench_args["preprocessed_data"],
        )
    elif task in QA_TASKS:
        if debug:
            file_path = TASK_TO_DEMO_FILE[task]
        else:
            file_path = TASK_TO_DEV_FILE[task]
        data = task_loader.load_qa(
            file_path,
            num_samples=num_samples,
            drop_duplicates=drop_duplicates
        )  # -> Dataset
    else:
        raise NotImplementedError(f"Task {task} is not supported.")

    assert isinstance(data, Dataset), "Data should be a Dataset object."

    return data


def retrieve(sample: dict,
             retriever: Retriever,
             no_retrieval: bool,
             zero_shot: bool,
             adaptive_retrieval: bool,
             return_true_k: Optional[bool]) -> Dict[str, Any]:
    """Retrieve documents using the retriever.
    
    Args:
        sample (dict): A single data sample containing 'query' and 'context'.
        retriever (Retriever): An instance of the Retriever class.
        no_retrieval (bool): If True, skip retrieval and use all documents.
        zero_shot (bool): If True, skip retrieval and do not use any documents.
        adaptive_retrieval (bool): If True, use adaptive retrieval; otherwise, use fixed-k retrieval.
        return_true_k (bool, optional): If True, return the true number of relevant documents.
        
    Returns:
        Dict[str, Any]: A dictionary containing the retrieval results.
    """
    query = sample["query"]
    context = sample["context"]

    if zero_shot:
        # print("Zero-shot setting enabled. No document will be retrieved.")
        retrieved_chunks = None
        true_k = None  # unknown
        retrieval_time = 0.0
        reduction_ratio = 0.0

    else:
        if no_retrieval:
            # print("No retrieval performed; all the documents will be included.")
            retriever.fixed_retrieval_k = len(sample["context"])
            retrieved_chunks, true_k = retriever.retrieve(
                query,
                context,
                sample=sample,
                adaptive=False,
                visualize=False,
                return_true_k=return_true_k
            )
            reduction_ratio = 0.0
            retrieval_time = 0.0

        else:  # RAG
            t = time.time()
            # if not adaptive retrieval, use fixed-k-chunks or fixed-n-tokens.
            # the fixed numbers are specified in the retriever class variables.
            retrieved_chunks, true_k = retriever.retrieve(
                query=query,
                context=context,
                sample=sample,
                adaptive=adaptive_retrieval,
                visualize=False,
                return_true_k=return_true_k
            )
            retrieval_time = time.time() - t
            reduction_ratio = 1 - len(retrieved_chunks) / \
                len(sample["context"])
            if reduction_ratio < 0.0:
                reduction_ratio = 0.0
            # print(f"Retrieval time: {retrieval_time:.2f} seconds")
            # print(
            #     f"Num retrieved docs: {0 if not retrieved_chunks else len(retrieved_chunks)}")
            # print(f"True k: {true_k}")
            # print(f"Reduction ratio: {reduction_ratio * 100:.2f}%")

    return {
        "question": query,
        "sample_idx": sample["idx"],
        "retrieved_docs": retrieved_chunks,
        "retrieval_time": retrieval_time,
        "num_retrieved_docs": str(0) if not retrieved_chunks else str(len(retrieved_chunks)),
        "reduction_ratio": reduction_ratio,
        "true_k": str(true_k),
        "positive_docs": sample["positive_context"]
    }  # str() is used to avoid ValueError: Circular reference detected when json.dump()


def generate(
    model: str,
    prompt: str,
    max_tokens: int = 16384,
    thinking_budget: int = 0,
    model_obj: Optional[PreTrainedModel] = None,
    tokenizer_obj: Optional[PreTrainedTokenizer] = None,
)  -> Dict[str, str | int]:
    """Ask the LLM using litellm or HF chat_request_hf.
    
    Args:
        model (str): Model name.
        prompt (str): Prompt to send to the model.
        max_tokens (int): Maximum number of tokens to generate.
        thinking_budget (int): Token budget for "thinking" (if supported).
        model_obj (PreTrainedModel, optional): Pre-loaded HF model object (for Qwen).
        tokenizer_obj (PreTrainedTokenizer, optional): Pre-loaded HF tokenizer object (for Qwen).
        
    Returns:
        Dict[str, str | int]: A dictionary containing the model's response and token usage information.
    """
    if "Qwen" in model:
        # Use the pre-loaded HF model and tokenizer
        if model_obj is None or tokenizer_obj is None:
             raise ValueError("Qwen model requires model_obj and tokenizer_obj to be provided.")
        try:
            response = chat_request_hf(
                model=model_obj,
                tokenizer=tokenizer_obj,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.0, # Fixed temperature for deterministic output
            )
            prediction: str = response.choices[0].message.content
            num_input_tokens: int = response.usage.prompt_tokens
            num_output_tokens: int = response.usage.completion_tokens
        except Exception as e:
            print(f"Error calling chat_request_hf for Qwen: {e}")
            prediction = ""
            # Fallback token count estimation
            num_input_tokens = utils.num_tokens_from_string(prompt, model_name=model)
            num_output_tokens = 0
    else:
        # Use litellm for other models
        if "o3-mini" in model:
            max_tokens = None
        if thinking_budget > 0:
            thinking_tag = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            thinking_tag = {"type": "disabled", "budget_tokens": 0}

        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                num_retries=10,
                vertex_credentials=vertex_credentials_json,
                thinking=thinking_tag,
                drop_params=True
            )

            prediction: str = response.choices[0].message.content
            num_input_tokens: int = response.usage.prompt_tokens
            num_output_tokens: int = response.usage.completion_tokens

        except Exception as e:
            print(f"Error calling litellm completion for {model}: {e}")
            prediction = ""
            num_input_tokens = utils.num_tokens_from_string(prompt, model_name=model)
            num_output_tokens = 0

    return {
        "prediction": prediction,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens
    }

async def generate_async(
    model: str,
    prompt: str,
    max_tokens: int = 16384,
    thinking_budget: int = 0,
    model_obj: Optional[PreTrainedModel] = None,
    tokenizer_obj: Optional[PreTrainedTokenizer] = None,
)  -> Dict[str, str | int]:
    """Ask the LLM asynchronously using litellm or HF chat_request_hf (wrapped).
    
    Args:
        model (str): Model name.
        prompt (str): Prompt to send to the model.
        max_tokens (int): Maximum number of tokens to generate.
        thinking_budget (int): Token budget for "thinking" (if supported).
        model_obj (PreTrainedModel, optional): Pre-loaded HF model object (for Qwen).
        tokenizer_obj (PreTrainedTokenizer, optional): Pre-loaded HF tokenizer object (for Qwen).
        
    Returns:
        Dict[str, str | int]: A dictionary containing the model's response and token usage information.
    """
    if "Qwen" in model:
        # Use the pre-loaded HF model and tokenizer, wrapped for async
        if model_obj is None or tokenizer_obj is None:
             raise ValueError("Qwen model requires model_obj and tokenizer_obj to be provided.")
        try:
            # Wrap the synchronous HF call in asyncio.to_thread
            response = await asyncio.to_thread(
                chat_request_hf,
                model=model_obj,
                tokenizer=tokenizer_obj,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.0, # Fixed temperature for deterministic output
            )
            prediction: str = response.choices[0].message.content
            num_input_tokens: int = response.usage.prompt_tokens
            num_output_tokens: int = response.usage.completion_tokens
        except Exception as e:
            print(f"Error calling chat_request_hf (async) for Qwen: {e}")
            prediction = ""
            # Fallback token count estimation
            num_input_tokens = utils.num_tokens_from_string(prompt, model_name=model)
            num_output_tokens = 0
    else:
        # Use litellm for other models
        if "o3-mini" in model:
            max_tokens = None
        if thinking_budget > 0:
            thinking_tag = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            thinking_tag = {"type": "disabled", "budget_tokens": 0}

        try:
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                num_retries=5,
                vertex_credentials=vertex_credentials_json,
                thinking=thinking_tag,
                drop_params=True
            )

            prediction: str = response.choices[0].message.content
            num_input_tokens: int = response.usage.prompt_tokens
            num_output_tokens: int = response.usage.completion_tokens

        except Exception as e:
            print(f"Error calling litellm acompletion for {model}: {e}")
            prediction = ""
            num_input_tokens = utils.num_tokens_from_string(prompt, model_name=model)
            num_output_tokens = 0

    return {
        "prediction": prediction,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens
    }


def rag(sample: Dict[str, Any],
        classify: bool,
        classifier_model: str,
        classifier_mode: str,
        classifier_prompt_template: str,
        structured_output: bool,
        retriever: Retriever,
        generation_model: str,
        max_tokens: int,
        thinking_budget: int,
        generation_prompt_template: str,
        no_retrieval: bool = False,
        zero_shot: bool = False,
        adaptive_retrieval: bool = True,
        return_true_k: bool = True,
        model_obj: Optional[PreTrainedModel] = None,
        tokenizer_obj: Optional[PreTrainedTokenizer] = None) -> Dict[str, Any]:
    """Run retrieval-augmented generation on the data.
    
    Args:
        sample (dict): A single data sample containing 'query' and 'context'.
        classify (bool): Whether to perform classification.
        classifier_model (str): The model to use for classification.
        classifier_mode (str): The mode to use for classification.
        classifier_prompt_template (str): The prompt template for classification.
        structured_output (bool): Whether to use structured output.
        retriever (Retriever): An instance of the Retriever class.
        generation_model (str): The model to use for generation.
        max_tokens (int): The maximum number of tokens to generate.
        thinking_budget (int): The token budget for "thinking" (if supported).
        generation_prompt_template (str): The prompt template for generation.
        no_retrieval (bool): If True, skip retrieval and use all documents.
        zero_shot (bool): If True, skip retrieval and do not use any documents.
        adaptive_retrieval (bool): If True, use adaptive retrieval; otherwise, use fixed-k retrieval.
        return_true_k (bool, optional): If True, return the true number of relevant documents.
        model_obj (Optional[PreTrainedModel]): Pre-loaded model object (if available).
        tokenizer_obj (Optional[PreTrainedTokenizer]): Pre-loaded tokenizer object (if available).

    Returns:
        Dict[str, Any]: A dictionary containing the retrieval results.
    """
    if classify:
        classification_result = classifier.classify(query=sample["query"],
                                                    model=classifier_model,
                                                    mode=classifier_mode,
                                                    prompt_template=classifier_prompt_template,
                                                    structured_output=structured_output,
                                                    temperature=0.0)
        category = classification_result["category"]
    else:
        category = None

    if (category is not None) and (category not in RETRIEVAL_CATEGORIES):
        # Got the answer already
        print("The query is answerable zero-shot:", category)
        print("Gold answer:", sample["label"])
        retrieval_result = {
            "question": sample["query"],
            "sample_idx": sample["idx"],
            "retrieved_docs": None,
            "retrieval_time": 0.0,
            "num_retrieved_docs": 0,
            "reduction_ratio": 100.0,
            "true_k": None,  # unknown
            "positive_docs": sample["positive_context"]
        }
        generation_result = {
            "question": sample["query"],
            "sample_idx": sample["idx"],
            "category": "zero-shot",
            "prediction": category,
            "gold": sample["label"],
            "num_input_tokens": classification_result["num_input_tokens"],
            "num_output_tokens": classification_result["num_output_tokens"],
            "reasoning_time": classification_result["reasoning_time"],
        }

        return {"retrieval_result": retrieval_result,
                "generation_result": generation_result}

    else:
        # retrieval
        # print(f"Starting retrieval for sample {sample['idx']}...")
        retrieval_result = retrieve(sample=sample,
                                    retriever=retriever,
                                    no_retrieval=no_retrieval,
                                    zero_shot=zero_shot,
                                    adaptive_retrieval=adaptive_retrieval,
                                    return_true_k=return_true_k)
        # retrieval_result["true_k"] = int(retrieval_result["true_k"])
        # retrieval_result["num_retrieved_docs"] = int(retrieval_result["num_retrieved_docs"])
        # print("Retrieval done.")

        if retrieval_result["retrieved_docs"] is None:
            retrieved_docs = None
        else:
            retrieved_docs = [doc["chunk"]
                              for doc in retrieval_result["retrieved_docs"]]

        # Generation
        # print(f"Starting reasoning for sample {sample['idx']}...")
        if zero_shot:
            assert retrieved_docs is None, \
                "No retrieval should be performed in zero-shot setting."
            if generation_prompt_template is None:
                prompt = sample["query"] # Use the query as the prompt
            else:
                prompt = generation_prompt_template.format(
                    question=sample["query"])
        else:
            assert generation_prompt_template is not None, \
                "Generation prompt template should be provided."

            # Use the retrieved documents as the prompt
            prompt = "\n\n".join(retrieved_docs)
            prompt = generation_prompt_template.format(
                context="\n\n".join(retrieved_docs),
                question=sample["query"]
            )

        t = time.time()
        prediction_dict = generate(generation_model,
                                   prompt,
                                   max_tokens,
                                   thinking_budget,
                                   model_obj=model_obj,
                                   tokenizer_obj=tokenizer_obj)
        reasoning_time = time.time() - t
        # print("Reasoning done.")
        # print(f"Reasoning time: {reasoning_time:.2f} seconds")

        generation_result = {
            "question": sample["query"],
            "sample_idx": sample["idx"],
            "category": category,
            "prediction": prediction_dict["prediction"],
            "gold": sample["label"],
            "num_input_tokens": prediction_dict["num_input_tokens"],
            "num_output_tokens": prediction_dict["num_output_tokens"],
            "reasoning_time": reasoning_time,
        }

        return {
            "retrieval_result": retrieval_result,
            "generation_result": generation_result
        }


async def rag_async(sample: Dict[str, Any],
                    classify: bool,
                    classifier_model: str,
                    classifier_mode: str,
                    classifier_prompt_template: str,
                    structured_output: bool,
                    retriever: Retriever,
                    generation_model: str,
                    max_tokens: int,
                    thinking_budget: int,
                    generation_prompt_template: str,
                    no_retrieval: bool = False,
                    zero_shot: bool = False,
                    adaptive_retrieval: bool = True,
                    return_true_k: bool = True,
                    model_obj: Optional[PreTrainedModel] = None,
                    tokenizer_obj: Optional[PreTrainedTokenizer] = None) -> Dict[str, Any]:
    """RAG with async OpenAI generation.
    
    Args:
        sample (dict): A single data sample containing 'query' and 'context'.
        classify (bool): Whether to perform classification.
        classifier_model (str): The model to use for classification.
        classifier_mode (str): The mode to use for classification.
        classifier_prompt_template (str): The prompt template for classification.
        structured_output (bool): Whether to use structured output.
        retriever (Retriever): An instance of the Retriever class.
        generation_model (str): The model to use for generation.
        max_tokens (int): The maximum number of tokens to generate.
        thinking_budget (int): The token budget for "thinking" (if supported).
        generation_prompt_template (str): The prompt template for generation.
        no_retrieval (bool): If True, skip retrieval and use all documents.
        zero_shot (bool): If True, skip retrieval and do not use any documents.
        adaptive_retrieval (bool): If True, use adaptive retrieval; otherwise, use fixed-k retrieval.
        return_true_k (bool, optional): If True, return the true number of relevant documents.
        model_obj (Optional[PreTrainedModel]): Pre-loaded model object (if available).
        tokenizer_obj (Optional[PreTrainedTokenizer]): Pre-loaded tokenizer object (if available).

    Returns:
        Dict[str, Any]: A dictionary containing the retrieval results.
    """
    # Run classifier (if requested) in a thread
    if classify:
        classification_result = await asyncio.to_thread(
            classifier.classify,
            query=sample["query"],
            model=classifier_model,
            mode=classifier_mode,
            prompt_template=classifier_prompt_template,
            structured_output=structured_output,
            temperature=0.0
        )
        category = classification_result["category"]
    else:
        category = None

    if (category is not None) and (category not in RETRIEVAL_CATEGORIES):
        # Got the answer already
        print("The query is answerable zero-shot:", category)
        print("Gold answer:", sample["label"])
        retrieval_result = {
            "question": sample["query"],
            "sample_idx": sample["idx"],
            "retrieved_docs": None,
            "retrieval_time": 0.0,
            "num_retrieved_docs": 0,
            "reduction_ratio": 100.0,
            "true_k": None,  # unknown
            "positive_docs": sample["positive_context"]
        }
        generation_result = {
            "question": sample["query"],
            "sample_idx": sample["idx"],
            "category": "zero-shot",
            "prediction": category,
            "gold": sample["label"],
            "num_input_tokens": classification_result["num_input_tokens"],
            "num_output_tokens": classification_result["num_output_tokens"],
            "reasoning_time": classification_result["reasoning_time"],
        }

        return {"retrieval_result": retrieval_result,
                "generation_result": generation_result}

    else:
        # print(f"Starting retrieval for sample {sample['idx']}...")
        # Wrap retrieval in a thread if it is CPU/blocking
        retrieval_result = await asyncio.to_thread(
            retrieve,
            sample=sample,
            retriever=retriever,
            no_retrieval=no_retrieval,
            zero_shot=zero_shot,
            adaptive_retrieval=adaptive_retrieval,
            return_true_k=return_true_k
        )
        # print("Retrieval done.")

        if retrieval_result["retrieved_docs"] is None:
            retrieved_docs = None
        else:
            retrieved_docs = [doc["chunk"]
                              for doc in retrieval_result["retrieved_docs"]]

        # Generation
        # print(f"Starting reasoning for sample {sample['idx']}...")
        if zero_shot:
            # Use the query as prompt if no retrieval was done
            assert retrieved_docs is None, \
                "No retrieval should be performed in zero-shot setting."
            if generation_prompt_template is None:
                # Use the query as the prompt
                prompt = sample["query"]
            else:
                prompt = generation_prompt_template.format(
                    question=sample["query"]
                )
        else:
            assert generation_prompt_template is not None, \
                "Generation prompt template should be provided."

            # Use the retrieved documents as the prompt
            prompt = "\n\n".join(retrieved_docs)
            prompt = generation_prompt_template.format(
                context="\n\n".join(retrieved_docs),
                question=sample["query"]
            )
        # Run generation asynchronously
        t = time.time()
        prediction_dict = await generate_async(
            generation_model,
            prompt,
            max_tokens,
            thinking_budget=thinking_budget,
            model_obj=model_obj,
            tokenizer_obj=tokenizer_obj
        )
        reasoning_time = time.time() - t
        # print("Reasoning done.")
        # print(f"Reasoning time: {reasoning_time:.2f} seconds")

        generation_result = {
            "question": sample["query"],
            "sample_idx": sample["idx"],
            "category": category,
            "prediction": prediction_dict["prediction"],
            "gold": sample["label"],
            "num_input_tokens": prediction_dict["num_input_tokens"],
            "num_output_tokens": prediction_dict["num_output_tokens"],
            "reasoning_time": reasoning_time,
        }

        return {
            "retrieval_result": retrieval_result,
            "generation_result": generation_result
        }


def pipeline(data: Dataset,
             classify: bool = False,
             classifier_model: str = "gpt-4o-mini",
             mode: str = "hybrid",
             classifier_prompt_template: str = None,
             structured_output: bool = True,
             retriever: Retriever = None,
             adaptive_retrieval: bool = True,
             return_true_k: bool = True,
             zero_shot: bool = False,
             no_retrieval: bool = False,
             generation_model: str = None,
             generation_prompt_template: str = None,
             max_tokens: int = None,
             thinking_budget: int = None,
             retrieval_results_file: str = "retrieval_results.json",
             generation_results_file: str = "generation_results.json",
             overwrite_results: bool = False,
             model_obj: Optional[PreTrainedModel] = None,
             tokenizer_obj: Optional[PreTrainedTokenizer] = None) -> None:
    """
    Classifier: hybrid →(if zero-shot) Answer
    ↓(if not zero-shot)
    Retriever: {fixed-k, adaptive-k}
    ↓
    LLM
    ↓
    Answer
    
    Args:
        data (Dataset): Dataset to process.
        classify (bool): Whether to perform classification.
        classifier_model (str): The model to use for classification.
        mode (str): The mode to use for classification.
        classifier_prompt_template (str): The prompt template for classification.
        structured_output (bool): Whether to use structured output.
        retriever (Retriever): An instance of the Retriever class.
        adaptive_retrieval (bool): If True, use adaptive retrieval; otherwise, use fixed-k retrieval.
        return_true_k (bool, optional): If True, return the true number of relevant documents.
        zero_shot (bool): If True, skip retrieval and do not use any documents.
        no_retrieval (bool): If True, skip retrieval and use all documents.
        generation_model (str): The model to use for generation.
        generation_prompt_template (str): The prompt template for generation.
        max_tokens (int): The maximum number of tokens to generate.
        thinking_budget (int): The token budget for "thinking" (if supported).
        retrieval_results_file (str): File to save retrieval results.
        generation_results_file (str): File to save generation results.
        overwrite_results (bool): If True, overwrite existing results files.
        model_obj (Optional[PreTrainedModel]): Pre-loaded model object (if available).
        tokenizer_obj (Optional[PreTrainedTokenizer]): Pre-loaded tokenizer object (if available).
    
    Returns:
        str: Message indicating where results are saved.
    """
    if not overwrite_results:
        if os.path.exists(retrieval_results_file):
            print(f"Results file {retrieval_results_file} already exists. "
                  "Use --overwrite_results to overwrite it.")
            with open(retrieval_results_file, "r") as f:
                retrieval_results = json.load(f)
            print(f"Loaded {len(retrieval_results)} retrieval results.")
        else:
            retrieval_results = []
            print(f"Results file {retrieval_results_file} does not exist. "
                  "Will create a new one.")

        if os.path.exists(generation_results_file):
            print(f"Results file {generation_results_file} already exists. "
                  "Use --overwrite_results to overwrite it.")
            with open(generation_results_file, "r") as f:
                generation_results = json.load(f)
            print(f"Loaded {len(generation_results)} generation results.")
        else:
            generation_results = []
            print(f"Results file {generation_results_file} does not exist. "
                  "Will create a new one.")

    else:
        print(
            f"Overwriting results files {retrieval_results_file} and {generation_results_file}.")
        retrieval_results = []
        generation_results = []

    for i, sample in tqdm(enumerate(data)):
        # Skip the sample if it is already in the results
        if not overwrite_results:
            if (
                sample["idx"] in [x["sample_idx"] for x in retrieval_results] and
                sample["idx"] in [x["sample_idx"]
                                for x in generation_results]
            ):
                print(
                    f"Sample {sample['idx']} already processed. Skipping...")
                continue

        # print(f"Processing sample {i + 1}/{len(data)}...")
        # RAG result: Dict[str, str | int | Any]
        rag_result = rag(
            sample=sample,
            classify=classify,
            classifier_model=classifier_model,
            classifier_mode=mode,
            classifier_prompt_template=classifier_prompt_template,
            structured_output=structured_output,
            retriever=retriever,
            generation_model=generation_model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            generation_prompt_template=generation_prompt_template,
            no_retrieval=no_retrieval,
            zero_shot=zero_shot,
            adaptive_retrieval=adaptive_retrieval,
            return_true_k=return_true_k,
            model_obj=model_obj,
            tokenizer_obj=tokenizer_obj
        )
        retrieval_results.append(rag_result["retrieval_result"])
        generation_results.append(rag_result["generation_result"])

    # Save the results
    with open(retrieval_results_file, "w") as f:
        json.dump(retrieval_results, f, indent=4,
                  default=utils.tensor_to_serializable)
    with open(generation_results_file, "w") as f:
        json.dump(generation_results, f, indent=4,
                  default=utils.tensor_to_serializable)

    print("Done.")
    print(
        f"Results saved to {retrieval_results_file} and {generation_results_file}.")


async def pipeline_async(data: Dataset,
                         classify: bool = False,
                         classifier_model: str = "gpt-4o-mini",
                         mode: str = "hybrid",
                         classifier_prompt_template: str = None,
                         structured_output: bool = True,
                         retriever: Retriever = None,
                         adaptive_retrieval: bool = True,
                         return_true_k: bool = True,
                         zero_shot: bool = False,
                         no_retrieval: bool = False,
                         generation_model: str = None,
                         generation_prompt_template: str = None,
                         max_tokens: int = None,
                         thinking_budget: int = None,
                         retrieval_results_file: str = "retrieval_results.json",
                         generation_results_file: str = "generation_results.json",
                         overwrite_results: bool = False,
                         model_obj: Optional[PreTrainedModel] = None,
                         tokenizer_obj: Optional[PreTrainedTokenizer] = None) -> str:
    """Asynchronous pipeline:
    Classifier -> Retriever -> Generation
    LLM generation for each sample is processed concurrently, limited by a semaphore.
    """
    if not overwrite_results:
        if os.path.exists(retrieval_results_file):
            print(f"Results file {retrieval_results_file} already exists. "
                  "Use --overwrite_results to overwrite it.")
            with open(retrieval_results_file, "r") as f:
                retrieval_results = json.load(f)
            print(f"Loaded {len(retrieval_results)} retrieval results.")
        else:
            retrieval_results = []
            print(f"Results file {retrieval_results_file} does not exist. "
                  "Will create a new one.")

        if os.path.exists(generation_results_file):
            print(f"Results file {generation_results_file} already exists. "
                  "Use --overwrite_results to overwrite it.")
            with open(generation_results_file, "r") as f:
                generation_results = json.load(f)
            print(f"Loaded {len(generation_results)} generation results.")
        else:
            generation_results = []
            print(f"Results file {generation_results_file} does not exist. "
                  "Will create a new one.")

    else:
        print(
            f"Overwriting results files {retrieval_results_file} and {generation_results_file}.")
        retrieval_results = []
        generation_results = []

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(10) # Limit to 10 concurrent tasks

    async def run_rag_with_semaphore(sem: asyncio.Semaphore, **kwargs):
        async with sem:
            return await rag_async(**kwargs)

    tasks = []
    for i, sample in enumerate(data):
        if not overwrite_results:
            if (
                sample["idx"] in [x["sample_idx"] for x in retrieval_results] and
                sample["idx"] in [x["sample_idx"] for x in generation_results]
            ):
                print(f"Sample {sample['idx']} already processed. Skipping...")
                continue

        print(f"Processing sample {i + 1}/{len(data)}...")
        # Schedule the task, wrapped to use the semaphore
        tasks.append(
            asyncio.create_task(
                run_rag_with_semaphore(
                    semaphore,
                    sample=sample,
                    classify=classify,
                    classifier_model=classifier_model,
                    classifier_mode=mode,
                    classifier_prompt_template=classifier_prompt_template,
                    structured_output=structured_output,
                    retriever=retriever,
                    adaptive_retrieval=adaptive_retrieval,
                    return_true_k=return_true_k,
                    zero_shot=zero_shot,
                    no_retrieval=no_retrieval,
                    generation_model=generation_model,
                    generation_prompt_template=generation_prompt_template,
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                    model_obj=model_obj,
                    tokenizer_obj=tokenizer_obj
                )
            )
        )

    # Wait for all tasks to complete.
    results = await asyncio.gather(*tasks)

    # Save the results
    for result in results:
        retrieval_results.append(result["retrieval_result"])
        generation_results.append(result["generation_result"])

    with open(retrieval_results_file, "w") as f:
        json.dump(retrieval_results, f, indent=4,
                  default=utils.tensor_to_serializable)
    with open(generation_results_file, "w") as f:
        json.dump(generation_results, f, indent=4,
                  default=utils.tensor_to_serializable)
    print("Done.")
    print(
        f"Results saved to {retrieval_results_file} and {generation_results_file}."
    )


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Pipeline for query classification and retrieval."
    )

    def int_or_float(value: int | float) -> int | float:
        """Custom type for argparse to accept int or float.
        
        Args:
            value (str): The input value as a string.
        Returns:
            int | float: The converted value as int or float.
        """
        try:
            # Try converting to int
            return int(value)
        except ValueError:
            try:
                # Try converting to float
                return float(value)
            except ValueError:
                raise argparse.ArgumentTypeError(f"{value} is not a valid int or float")

    # General
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--classify",
        action="store_true",
        help="Run the classifier.",
    )
    general_group.add_argument(
        "--run_async",
        action="store_true",
        help="Run the pipeline asynchronously.",
    )

    # Data loading
    data_group = parser.add_argument_group("Data loading")
    data_group.add_argument(
        "--task",
        type=str,
        choices=SUPPORTED_TASKS,
        default="hotpotqa",
        help="Task to run.",
    )
    data_group.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to load.",
    )
    data_group.add_argument(
        "--drop_duplicates",
        action="store_true",
        help="Drop duplicates in the data.",
    )
    data_group.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode. Load one sample.",
    )

    # Holobench-related
    holobench_group = parser.add_argument_group("Holobench")
    holobench_group.add_argument(
        "--holobench_split",
        type=str,
        default="wine_1",
        choices=HOLOBENCH_SPLITS,
        help="Split to load from Holobench.",
    )
    holobench_group.add_argument(
        "--max_context_size",
        type=int,
        default=4096,
        help="Max context size for Holobench.",
    )
    holobench_group.add_argument(
        "--info_amount",
        type=float,
        default=2048,
        help="Information amount for Holobench.",
    )
    holobench_group.add_argument(
        "--merge_strategy",
        type=str,
        default="uniform",
        choices=MERGE_STRATEGIES,
        help="Merge strategy for Holobench.",
    )
    holobench_group.add_argument(
        "--preprocessed_data",
        type=str,
        default=None,
        help="Preprocessed data path. example: `dataset_src/holobench_5k.dataset`",
    )

    # Classifier
    classifier_group = parser.add_argument_group("Classifier")
    classifier_group.add_argument(
        "--classifier_model",
        type=str,
        default="gpt-4o-mini",
        help="Classifier model name.",
    )
    classifier_group.add_argument(
        "--classifier_mode",
        type=str,
        choices=["binary", "hybrid"],
        default="hybrid",
        help="Classifier mode.",
    )
    classifier_group.add_argument(
        "--structured_output",
        action="store_true",
        help="Use structured output for the classifier.",
    )
    classifier_group.add_argument(
        "--classifier_prompt_template",
        type=str,
        default="./templates/classifier_hybrid_template.txt",
        help="Prompt template file path for the classifier.",
    )
    classifier_group.add_argument(
        "--classifier_examples",
        type=str,
        default="./templates/classifier_examples.json",
        help="Prompt examples file path for the classifier.",
    )

    # Retriever
    retriever_group = parser.add_argument_group("Retriever")
    retriever_group.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Retriever model name.",
    )
    retriever_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for the retriever.",
    )
    retriever_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the retriever.",
    )
    retriever_group.add_argument(
        "--adaptive_retrieval",
        action="store_true",
        help="Use adaptive k for the retriever.",
    )
    retriever_group.add_argument(
        "--retrieval_strategy",
        type=str,
        default="largest_gap",
        choices=["largest_gap", "moving_avg", "2diff_spike"],
        help="Retrieval strategy for the retriever.",
    )
    retriever_group.add_argument(
        "--ignore_extreme",
        type=int_or_float,
        default=0.0,
        help="Ignore the top ignore_extreme * 100 percent of the extreme values in the retrieval strategy.",
    )
    retriever_group.add_argument(
        "--ignore_extreme_tail",
        type=int_or_float,
        default=0.0,
        help="Ignore the last ignore_extreme_tail * 100 percent of the extreme values in the retrieval strategy.",
    )
    retriever_group.add_argument(
        "--retrieve_more",
        type=int,
        help="Use more documents for the retriever by x (float) percent or x (int) docs.",
    )
    retriever_group.add_argument(
        "--fixed_retrieval_k",
        type=int,
        default=None,
        help=(
            "Fixed k for the retriever. "
            "If `args.adaptive_retrieval` is set to `True`, this will be ignored.",
        )
    )
    retriever_group.add_argument(
        "--fixed_retrieval_tokens",
        type=int,
        default=None,
        help=(
            "Fixed tokens for the retriever. "
            "If `args.adaptive_retrieval` is set to `True`, this will be ignored.",
        )
    )
    retriever_group.add_argument(
        "--use_sentence_transformer",
        action="store_true",
        help="Use sentence transformer for the retriever.",
    )
    retriever_group.add_argument(
        "--return_true_k",
        action="store_true",
        help="Return the true k for the retriever.",
    )
    retriever_group.add_argument(
        "--zero_shot",
        action="store_true",
        help="No retrieval, zero-shot.",
    )
    retriever_group.add_argument(
        "--full_context",
        action="store_true",
        help="No retrieval, all the documents will be included.",
    )
    retriever_group.add_argument(
        "--use_faiss",
        action="store_true",
        help="Use the premade FAISS index for the retriever.",
    )
    retriever_group.add_argument(
        "--compute_cossim_from_scratch",
        action="store_true",
        help="Compute the embedding similarities with SentenceTransformers (not FAISS).",
    )

    # Generation LLM
    generation_group = parser.add_argument_group("Generation LLM")
    generation_group.add_argument(
        "--generation_model",
        type=str,
        default="gpt-4o-mini",
        help="Generation model name.",
    )
    generation_group.add_argument(
        "--hf_generation_device",
        type=str,
        default="auto",
        help="Device to use for Hugging Face generation models (e.g., 'auto', 'cpu', 'cuda:0').",
    )
    generation_group.add_argument(
        "--generation_prompt_template",
        type=str,
        required=True,
        help="Prompt template file path for the generation model.",
    )
    generation_group.add_argument(
        "--max_generation_tokens",
        type=int,
        default=MAX_GENERATION_TOKENS,
        help="Max tokens for the generation model.",
    )
    generation_group.add_argument(
        "--thinking_budget",
        type=int,
        default=0,
        help="Set the thinking budget for the generation model.",
    )

    # Stats
    stats_group = parser.add_argument_group("Stats")
    stats_group.add_argument(
        "--retrieval_results_file",
        type=str,
        default="retrieval_results.json",
        help="File to save the retrieval results.",
    )
    stats_group.add_argument(
        "--generation_results_file",
        type=str,
        default="generation_results.json",
        help="File to save the generation results.",
    )
    stats_group.add_argument(
        "--overwrite_results",
        action="store_true",
        help="Overwrite the results files.",
    )

    return parser.parse_args()


def main() -> None:
    """Main function to run the pipeline."""
    args = get_args()
    print(args)

    logger.info("Loading the data...")
    data = load_data(
        task=args.task,
        num_samples=args.num_samples,
        drop_duplicates=args.drop_duplicates,
        debug=args.debug,
        holobench_split=args.holobench_split,
        max_context_size=args.max_context_size,
        info_amount=args.info_amount,
        merge_strategy=args.merge_strategy,
        preprocessed_data=args.preprocessed_data,
    )

    # Print the loaded data
    logger.info(f"Loaded {len(data)} samples from {args.task}.")

    # Load HF model and tokenizer if Qwen is specified
    model_obj = None
    tokenizer_obj = None
    # if "Qwen"  in args.generation_model:
    #     logger.info(f"Loading Hugging Face model: {args.generation_model} on device: {args.hf_generation_device}")
    #     tokenizer_obj = AutoTokenizer.from_pretrained(args.generation_model)
    #     model_obj = AutoModelForCausalLM.from_pretrained(
    #         args.generation_model,
    #         torch_dtype="auto", # Often recommended over float16
    #         device_map=args.hf_generation_device, # Use argument for device
    #         )
    #     logger.info("Hugging Face model loaded.")

    # A data sample (dict) should consist of:
    # 'query', 'context', 'answer', 'positive_context', and 'relevant_context'.

    # Instantiate the retriever
    # load BAAI models from HuggingFace
    hf = True if "BAAI" in args.retriever_model or "Alibaba-NLP" in args.retriever_model else False

    retriever = Retriever(model=args.retriever_model,
                          hf=hf,
                          use_sentence_transformer=args.use_sentence_transformer,
                          batch_size=args.batch_size,
                          device=args.device,
                          strategy=args.retrieval_strategy,
                          ignore_extreme=args.ignore_extreme,
                          ignore_extreme_tail=args.ignore_extreme_tail,
                          retrieve_more=args.retrieve_more,
                          fixed_retrieval_k=args.fixed_retrieval_k,
                          fixed_retrieval_tokens=args.fixed_retrieval_tokens,
                          task=args.task,
                          use_faiss=args.use_faiss,
                          compute_cossim_from_scratch=args.compute_cossim_from_scratch,
                          info_amount=args.info_amount,
                          use_gpu=(args.device == "cuda"),)


    if args.classifier_prompt_template is not None:
        with open(args.classifier_prompt_template, "r") as f:
            classifier_prompt_template = f.read()

        # Read and add the examples
        if args.classifier_examples is not None:
            examples_str = "\n\n## Examples\n"

            # Read
            with open(args.classifier_examples, "r") as f:
                classifier_examples: list = json.load(f)

            # take 10 random examples from classifier_examples
            zero_shot_examples = [
                ex for ex in classifier_examples if ex["expected_answer"] not in {"factoid", "aggregation"}
            ]
            factoid_examples = [
                ex for ex in classifier_examples if ex["expected_answer"] == "factoid"
            ]
            aggregation_examples = [
                ex for ex in classifier_examples if ex["expected_answer"] == "aggregation"
            ]
            # Randomly sample 3 examples from each category
            classifier_examples = random.sample(zero_shot_examples, 3) + \
                random.sample(factoid_examples, 3) + \
                random.sample(aggregation_examples, 3)
            # Shuffle the examples
            random.shuffle(classifier_examples)

            # Add the examples to the prompt
            for i, ex in enumerate(classifier_examples):
                # ex: Dict[str, str]
                examples_str += f"**Example {i + 1}**\n"
                examples_str += f"Question: {ex['question']}\n"
                examples_str += f"Expected answer: {ex['expected_answer']}\n"
                examples_str += f"Reason: {ex['reason']}\n\n"

            # Add the examples to the prompt
            classifier_prompt_template += examples_str

    # Generation template
    if args.task == "holobench":
        # Holobench template
        with open(args.generation_prompt_template, "r") as f:
            generation_prompt_template = f.read()

    elif args.task in QA_TASKS:
        assert args.generation_prompt_template is not None, \
            "Generation prompt template should be provided."

        with open(args.generation_prompt_template, "r") as f:
            generation_prompt_template = f.read()

    else:
        raise NotImplementedError

    if args.run_async:
        # Run the pipeline asynchronously
        asyncio.run(
            pipeline_async(
                data=data,
                classify=args.classify,
                classifier_model=args.classifier_model,
                mode=args.classifier_mode,
                classifier_prompt_template=classifier_prompt_template,
                structured_output=args.structured_output,
                retriever=retriever,
                adaptive_retrieval=args.adaptive_retrieval,
                return_true_k=args.return_true_k,
                zero_shot=args.zero_shot,
                no_retrieval=args.full_context,
                generation_model=args.generation_model,
                generation_prompt_template=generation_prompt_template,
                max_tokens=args.max_generation_tokens,
                thinking_budget=args.thinking_budget,
                retrieval_results_file=args.retrieval_results_file,
                generation_results_file=args.generation_results_file,
                overwrite_results=args.overwrite_results,
                model_obj=model_obj,
                tokenizer_obj=tokenizer_obj
            )
        )

    else:
        pipeline(data=data,
                 classify=args.classify,
                 classifier_model=args.classifier_model,
                 mode=args.classifier_mode,
                 classifier_prompt_template=classifier_prompt_template,
                 structured_output=True,
                 retriever=retriever,
                 adaptive_retrieval=args.adaptive_retrieval,
                 return_true_k=args.return_true_k,
                 zero_shot=args.zero_shot,
                 no_retrieval=args.full_context,
                 generation_model=args.generation_model,
                 generation_prompt_template=generation_prompt_template,
                 max_tokens=args.max_generation_tokens,
                 thinking_budget=args.thinking_budget,
                 retrieval_results_file=args.retrieval_results_file,
                 generation_results_file=args.generation_results_file,
                 overwrite_results=args.overwrite_results,
                 model_obj=model_obj,
                 tokenizer_obj=tokenizer_obj
                 )

    logger.info("DONE.")


if __name__ == "__main__":
    main()
