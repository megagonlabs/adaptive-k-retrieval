from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
import os
import json
import multiprocessing as mp
import torch
import dotenv
import asyncio
from openai import OpenAI, AsyncOpenAI
import openai

import argparse
from typing import List, Generator, Any
from tqdm import tqdm

# local
from solve import QA_TASKS, HOLOBENCH_TASKS


dotenv.load_dotenv()


EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/llm-embedder"
}

OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
}


def load_data(data_path: str) -> Dataset:
    """Load data from a jsonl file.

    Args:
        data_path (str): path to the jsonl file.

    Returns:
        list: list of Documents.
    """
    if data_path.endswith(".jsonl"):
        dataset = load_dataset(
            "json",
            data_files=data_path
        )["train"]
    elif data_path.endswith(".dataset"):
        dataset = load_from_disk(data_path)
        
    return dataset


def get_embedding(model: SentenceTransformer,
                  texts: List[str]) -> list:
    embeddings = model.encode(texts,
                              convert_to_numpy=True,  # This option is necessary to store in FAISS
                              show_progress_bar=False,
                              batch_size=64)
    return embeddings


def get_openai_embedding(model: str,
                         client: OpenAI,
                         texts: List[str]) -> list:
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        model=model,
        input=texts
    )

    return [e.embedding for e in response.data]


def chunks(lst: List[Any],
           batch_size: int) -> Generator[List[Any], None, None]:
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


async def async_get_openai_embedding(model: str,
                                     client: AsyncOpenAI,
                                     texts: List[str],
                                     batch_size: int = 64) -> list:
    """Get embedding from OpenAI API asynchronously."""
    all_embeddings = []
    
    # Process the texts in batches, so that they fit within the token limit (8192)
    for batch in chunks(texts, batch_size):
        batch = [" " if t == "" else  t for t in batch]
        # sometimes there can be empty strings but openai doesn't accept them
        
        while True:
            try:
                response = await client.embeddings.create(
                    model=model,
                    input=batch
                )
                break
            except openai.RateLimitError as e:
                print(e)
                await asyncio.sleep(2)
        
        # verify that the embeddings in the batch are in the same order as the input
        all_embeddings.extend([e.embedding for e in response.data])
    return [e.embedding for e in response.data]


async def process_sample(i: int,
                         sample: dict,
                         args: argparse.Namespace,
                         client: AsyncOpenAI,
                         output_dir: str,
                         semaphore: asyncio.Semaphore) -> None:
    """Process a single sample (OpenAI embeddings) asynchronously.
    
    Args:
        i (int): index of the sample.
        sample (dict): a single sample from the dataset.
        args (argparse.Namespace): command line arguments.
        client (AsyncOpenAI): OpenAI client.
        output_dir (str): path to the output directory.
        semaphore (asyncio.Semaphore): semaphore to limit concurrency.
    """
    async with semaphore:  # limit concurrency if needed
        destination = os.path.join(
            output_dir, f"{args.task}_contexts_{i}.index"
        )
        if os.path.exists(destination):
            print("Index already exists. Skipping...")
            return
        
        # Extract the contexts from the sample
        ctxs = [ctx["text"] for ctx in sample["ctxs"]]
        embeddings = await async_get_openai_embedding(args.model,
                                                      client,
                                                      ctxs)
        # Convert list of embeddings to numpy array (faiss requires numpy's float32)
        embeddings = np.array(embeddings, dtype=np.float32)
        index = create_index(embeddings)

        # Save the index to a file
        faiss.write_index(index, destination)
        print("Saved index to", destination)


async def run_async(dataset: Dataset,
                    args: argparse.Namespace,
                    client: AsyncOpenAI,
                    output_dir: str) -> None:
    """Run the async process for the entire dataset.
    
    Args:
        dataset (Dataset): the dataset to process.
        args (argparse.Namespace): command line arguments.
        client (AsyncOpenAI): OpenAI client.
        output_dir (str): path to the output directory.
    """
    # Optionally limit the number of concurrent API calls;
    # adjust the limit if needed.
    semaphore = asyncio.Semaphore(args.semaphore)

    # Create a task for each dataset sample.
    tasks = [
        process_sample(i,
                       sample,
                       args,
                       client=client,
                       output_dir=output_dir,
                       semaphore=semaphore)
        for i, sample in enumerate(dataset)
    ]
    
    progress_bar = tqdm(total=len(tasks), desc="Processing samples")
    for task in asyncio.as_completed(tasks):
        await task
        progress_bar.update(1)
    progress_bar.close()


def create_index(embeddings: np.ndarray) -> faiss.Index:
    """Create a FAISS index from embeddings.
    
    Args:
        embeddings (np.ndarray): numpy array of shape (num_vectors, embedding_dim)
    Returns:
        faiss.Index: FAISS index
    """
    embedding_dim = embeddings.shape[1]  # dimensionality of the embeddings
    # Normalize the embeddings to unit length
    faiss.normalize_L2(embeddings)  # for cosine similarity
    # IndexFlatIP for cosine simialrity
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    return index


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Make a vector database beforehand."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        choices=EMBEDDING_MODELS,
        help="The model to use for embedding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The device to use for embedding.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=set(QA_TASKS) | HOLOBENCH_TASKS,
        required=True,
        help="Task name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vector_dbs",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--run_async",
        action="store_true",
        help="Use async OpenAI API.",
    )
    parser.add_argument(
        "--semaphore",
        type=int,
        default=10,
        help="Number of concurrent API calls.",
    )

    return parser.parse_args()


def process_group(gpu_id: int,
                  tasks: list,
                  model_name: str,
                  output_dir: str,
                  task_name: str,
                  device: str):
    """Process a group of samples on a specific GPU.
    Each element in tasks is a tuple (i, sample).
    
    Args:
        gpu_id (int): GPU id to use.
        tasks (list): list of tuples (i, sample).
        model_name (str): name of the embedding model.
        output_dir (str): path to the output directory.
        task_name (str): name of the task.
        device (str): device to use ("cuda" or "cpu").
    """
    dev = f"cuda:{gpu_id}" if device == "cuda" else "cpu"
    model = SentenceTransformer(model_name).to(dev)
    print(f"Worker on device {dev} processing {len(tasks)} samples...")

    for i, sample in tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        if task_name in QA_TASKS:
            ctxs = [ctx["text"] for ctx in sample["ctxs"]]
        elif task_name in HOLOBENCH_TASKS:
            ctxs = [ctx["text"] for ctx in sample["context"]]
            
        embeddings = get_embedding(model, ctxs)
        index = create_index(embeddings)
        destination = os.path.join(
            output_dir, f"{task_name}_contexts_{i}.index")
        faiss.write_index(index, destination)
        print("Saved index to", destination)


def single_process_qa(dataset: Dataset,
                      model: SentenceTransformer,
                      output_dir: str,
                      task: str) -> None:
    """Process a single dataset for QA tasks.
    
    Args:
        dataset (Dataset): the dataset to process.
        model (SentenceTransformer): the embedding model.
        output_dir (str): path to the output directory.
        task (str): name of the task.
    """
    for i, sample in tqdm(enumerate(dataset)):
        ctxs = [ctx["text"] for ctx in sample["ctxs"]]
        embeddings = get_embedding(model, ctxs)
        index = create_index(embeddings)
        destination = os.path.join(output_dir, f"{task}_contexts_{i}.index")
        faiss.write_index(index, destination)
        print("Saved index to", destination)


if __name__ == "__main__":
    args = get_args()

    if args.num_gpus >= 1:
        assert torch.cuda.is_available(), "CUDA is not available. Please use CPU."
    if args.task in QA_TASKS:
        pass
    elif args.task in HOLOBENCH_TASKS:
        pass
    else:
        raise ValueError(
            f"Task {args.task} not supported. Supported tasks are {QA_TASKS}."
        )

    if args.device == "cuda":
        mp.set_start_method("spawn", force=True)

    # load the data
    with open("task_to_dataset.json", "r") as f:
        task_to_dataset = json.load(f)
    data_path = task_to_dataset[args.task]
    dataset = load_data(data_path)

    data_name = os.path.splitext(os.path.basename(data_path))[0]

    # output file should be:
    # vector_dbs/{args.task}/{args.model.split("/")}contexts_0.index
    output_dir = os.path.join(args.output_dir, args.task)
    output_dir = os.path.join(output_dir, args.model.split("/")[-1])
    # ignores "BAAI", while retaining OpenAI model names intact
    os.makedirs(output_dir, exist_ok=True)

    if "BAAI" in args.model:
        model = SentenceTransformer(args.model).to(args.device)
        if args.device == "cuda":
            if args.num_gpus < 1:
                print("Warning: num_gpus is less than 1. Using CPU instead.")
                num_workers = 1
                args.device = "cpu"
            else:
                print(f"Using {args.num_gpus} GPUs.")
                num_workers = args.num_gpus
        else:  # CPU
            num_workers = 1

        groups = {i: [] for i in range(num_workers)}
        for idx, sample in enumerate(dataset):
            worker_id = idx % num_workers
            groups[worker_id].append((idx, sample))
        processes = []
        for worker_id, tasks in groups.items():
            # Launch a process for each worker group.
            p = mp.Process(
                target=process_group,
                args=(worker_id, tasks, args.model,
                      output_dir, args.task, args.device)
            )
            p.start()
            processes.append(p)
        # Wait for all processes to finish.
        for p in processes:
            p.join()

    elif args.model in OPENAI_EMBEDDING_MODELS:
        if args.run_async:
            client = AsyncOpenAI()
            asyncio.run(run_async(dataset,
                                  args,
                                  client=client,
                                  output_dir=output_dir))
        else:
            client = OpenAI()

            for i, sample in enumerate(dataset):
                ctxs = [ctx["text"] for ctx in sample["ctxs"]]
                embeddings = get_openai_embedding(args.model, client, ctxs)
                index = create_index(embeddings)
                destination = os.path.join(
                    output_dir, f"{args.task}_contexts_{i}.index")
                faiss.write_index(index, destination)
                print("Saved index to", destination)
        
        print("Done.")
        print(f"Vector database saved in {output_dir}.")

    else:
        raise NotImplementedError
