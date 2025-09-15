from typing import List, Dict, Tuple, Optional, Literal
import faiss
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch

# local
import utils
from eval import ContextType


# Load the config file
with open("config.json", "r") as f:
    config = json.load(f)
base_dir = config["adaptive_rag_dir"]


CHAR_TO_TOKEN_RATIO = 3.66 # defined in the HoloBench paper
HOLOBENCH_TASKS = {
    "holobench_1k",
    "holobench_5k",
    "holobench_10k",
    "holobench_25k",
    "holobench_50k",
}


class Retriever:
    def __init__(self,
                 model: str = "BAAI/bge-large-en-v1.5",
                 hf: bool = True,
                 use_sentence_transformer: bool = True,
                 batch_size: int = 1,
                 device: str = "cpu",
                 strategy: Optional[str] = None,
                 window: Optional[int] = None,
                 retrieve_more: Optional[int | float] = None,
                 ignore_extreme: Optional[float] = None,
                 ignore_extreme_tail: Optional[float] = None,
                 ignore_below_median: Optional[bool] = None,
                 use_faiss: Optional[bool] = None,
                 compute_cossim_from_scratch: Optional[bool] = None,
                 fixed_retrieval_k: Optional[int] = None,
                 fixed_retrieval_tokens: Optional[int] = None,
                 task: str = None,
                 info_amount: Optional[int] = None,
                 use_gpu: bool = False):
        """Retriever class.
        Args:
            model (str): The model name.
            hf (bool): If True, use Hugging Face model.
            use_sentence_transformer (bool): If True, use SentenceTransformer.
            batch_size (int): Batch size for embedding.
            device (str): Device to use (e.g., "cpu", "cuda").
            strategy (str): Strategy for finding the threshold.
                Options: "largest_gap", "moving_avg", "2diff_spike".
            window (int): Window size for moving average.
            retrieve_more (int | float): If int, retrieve more chunks.
                If float, multiply the threshold by this value.
            ignore_extreme (float): Ignore extreme values.
            ignore_extreme_tail (float): Ignore extreme values at the tail.
            ignore_below_median (bool): If True, ignore values below the median.
            use_faiss (bool): If True, use FAISS for retrieval.
            compute_cossim_from_scratch (bool): If True, compute cosine similarity from scratch.
            fixed_retrieval_k (int): Fixed number of chunks to retrieve.
            fixed_retrieval_tokens (int): Fixed number of tokens to retrieve.
            task (str): Task name.
            info_amount (int): Number of chunks to retrieve.
            use_gpu (bool): If True, use GPU for retrieval.
        """
        self.model = model
        self.hf = hf
        self.use_sentence_transformer = use_sentence_transformer
        self.batch_size = batch_size
        self.device = device
        self.strategy = strategy
        self.window = window
        self.retrieve_more = 0 if retrieve_more is None else retrieve_more
        self.ignore_extreme = 0.0 if ignore_extreme is None else ignore_extreme
        self.ignore_extreme_tail = 0.0 if ignore_extreme_tail is None else ignore_extreme_tail
        self.ignore_below_median = ignore_below_median
        self.use_faiss = use_faiss # If True, the pre-made context embeddings will be used.
        self.compute_cossim_from_scratch = compute_cossim_from_scratch # compute the embedding similarities with SentenceTransformer
        self.fixed_retrieval_k = fixed_retrieval_k
        self.fixed_retrieval_tokens = fixed_retrieval_tokens
        self.char_to_token_ratio = 3.3  # rough estimate, following the HoloBench paper
        self.task = task
        self.info_amount = info_amount # for holobench
        self.use_gpu = use_gpu
        if self.use_gpu and self.use_faiss:
            raise NotImplementedError("faiss-gpu has not been supported in this code yet.")

        if hf:
            self.embedding = utils.Embedding(self.model,
                                             self.hf,
                                             use_sentence_transformer=use_sentence_transformer,
                                             device=device)
        else:
            raise NotImplementedError  # TODO: GPT

    def _create_vector_database(self,
                                context_chunks: list,
                                index_name: str = "vector_index.index",
                                mode: Literal["IP", "L2"] = "IP") -> faiss.Index:
        """DEPRECATED.
        Method for creating a faiss index from a list of context chunks.
        """
        embeddings = self.embedding.get_embedding(context_chunks)
        embedding_matrix = np.vstack(embeddings)

        if mode == "IP":
            embedding_matrix = embedding_matrix / \
                np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            # Initialize FAISS index
            dimension = embedding_matrix.shape[1]  # embedding dimension
            index = faiss.IndexFlatIP(dimension)
        elif mode == "L2":
            dimension = embedding_matrix.shape[1]  # embedding dimension
            index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(
                (f"{mode} is not supported; "
                 "it should be either `IP` or `L2` (`IP` is recommended)")
            )

        index.add(embedding_matrix)

        # Save the vector database index
        faiss.write_index(index, index_name)
        print(f"Vector database created and saved in {index_name}.")

        return index

    def _get_cosine_similarities(self,
                                 query_emb: List[float],
                                 index: faiss.IndexFlatIP) -> List[float]:
        """DEPRECATED.
        Get the cosine similarity between a query (embedding) and
        the FAISS index containing context embeddings.

        Args:
            query_emb (list[float]): A query embedding.
            index (faiss.IndexFlatIP): faiss index.
        Returns:
            List[float]: A list of cosine similarities
        """
        # Normalize the query embedding
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.flatten()  # (1, 1024) -> (1024, )

        # Retrieve all the context embeddings from the FAISS index
        index_embs = index.reconstruct_n(0, index.ntotal)

        # Normalize the context embeddings <- maybe not necessary?
        index_norms = np.linalg.norm(index_embs, axis=1, keepdims=True)
        index_embs = index_embs / index_norms

        cosine_similarities = np.dot(index_embs, query_emb)

        return cosine_similarities.tolist()

    def _is_ascending(self, arr: np.ndarray | torch.Tensor) -> bool:
        """Check if the list is sorted in ascending order.
        Args:
            arr (np.ndarray | torch.Tensor): A list of cosine similarities.
        Returns:
            bool: True if the list is sorted in ascending order, False otherwise.
        """
        # Check if all differences are <= 0
        if isinstance(arr, torch.Tensor):
            return torch.all(torch.diff(arr) >= 0)
        elif isinstance(arr, np.ndarray):
            return np.all(np.diff(arr) >= 0)
        else:
            raise TypeError(
                "Invalid type; use either np.ndarray or torch.Tensor.")

    def _is_descending(self, arr: np.ndarray | torch.Tensor) -> bool:
        """Check if the list is sorted in descending order.
        Args:
            arr (np.ndarray | torch.Tensor): A list of cosine similarities.
        Returns:
            bool: True if the list is sorted in descending order, False otherwise.
        """
        # Check if all differences are <= 0
        if isinstance(arr, torch.Tensor):
            return torch.all(torch.diff(arr) <= 0)
        elif isinstance(arr, np.ndarray):
            return np.all(np.diff(arr) <= 0)
        else:
            raise TypeError(
                "Invalid type; use either np.ndarray or torch.Tensor.")

    def find_threshold_largest_gap(self, arr: torch.Tensor) -> int:
        """Find the threshold using the largest gap method.
        Args:
            arr (torch.Tensor): A list of cosine similarities.
        Returns:
            int: The index where the largest numerical gap occurs.
        """
        if self.ignore_below_median:
            self.ignore_extreme_tail = 0.5

        if isinstance(self.ignore_extreme, float):
            cut_index = int((len(arr) - 1) * self.ignore_extreme)
        elif isinstance(self.ignore_extreme, int):
            cut_index = self.ignore_extreme
        
        if isinstance(self.ignore_extreme_tail, float):
            cut_index_tail = int((len(arr) - 1) * self.ignore_extreme_tail)
        elif isinstance(self.ignore_extreme_tail, int):
            cut_index_tail = self.ignore_extreme_tail

        # Calculate the gaps between consecutive elements
        gaps = torch.diff(arr)

        # Find the (first) index of the largest gap
        if cut_index_tail == 0:
            threshold = torch.argmin(gaps[cut_index:]) + cut_index
        else:
            threshold = torch.argmin(
                gaps[cut_index:-cut_index_tail]) + cut_index

        return threshold

    def find_threshold_moving_avg(self, arr: torch.Tensor) -> int:
        """Find the threshold using the moving average method.
        Args:
            arr (torch.Tensor): A list of cosine similarities.
        Returns:
            int: The index where the largest numerical gap occurs.
        """
        assert self.window is not None, \
            "`window` must be specified when using `moving_avg`."

        if self.ignore_below_median:
            self.ignore_extreme_tail = 0.5

        if isinstance(self.ignore_extreme, float):
            cut_index = int((len(arr) - 1) * self.ignore_extreme)
        elif isinstance(self.ignore_extreme, int):
            cut_index = self.ignore_extreme
            
        if isinstance(self.ignore_extreme_tail, float):
            cut_index_tail = int((len(arr) - 1) * self.ignore_extreme_tail)
        elif isinstance(self.ignore_extreme_tail, int):
            cut_index_tail = self.ignore_extreme_tail
            
        # Calculate the moving average
        kernel = torch.ones(self.window, device=arr.device) / self.window
        moving_avg = torch.conv1d(arr.view(1, 1, -1),
                                  kernel.view(1, 1, -1),
                                  padding=0).squeeze()
        # Calculate the gaps between consecutive elements
        gaps = torch.diff(moving_avg)
        
        # Find the (first) index of the largest gap
        if cut_index_tail == 0:
            threshold = torch.argmin(gaps[cut_index:]) + cut_index
        else:
            threshold = torch.argmin(
                gaps[cut_index:-cut_index_tail]) + cut_index
        return threshold

    def find_threshold_2diff_spike(self, arr: torch.Tensor) -> int:
        """Find the threshold using the second derivative spike method.
        Args:
            arr (torch.Tensor): A list of cosine similarities.
        Returns:
            int: The index where the largest numerical gap occurs.
        """
        if self.ignore_below_median:
            self.ignore_extreme_tail = 0.5

        # will be 0 if ignore_extreme = 0.0
        if isinstance(self.ignore_extreme, float):
            cut_index = int((len(arr) - 1) * self.ignore_extreme)
        elif isinstance(self.ignore_extreme, int):
            cut_index = self.ignore_extreme
            
        # will be 0 if ignore_extreme_tail = 0.0
        if isinstance(self.ignore_extreme_tail, float):
            cut_index_tail = int((len(arr) - 1) * self.ignore_extreme_tail)
        elif isinstance(self.ignore_extreme_tail, int):
            cut_index_tail = self.ignore_extreme_tail

        # cut the extreme values
        if cut_index_tail == 0:
            arr = arr[cut_index:]
        else:
            arr = arr[cut_index:-cut_index_tail]

        # Calculate the second derivative
        first_diff = torch.diff(arr)
        second_diff = torch.diff(first_diff)

        cum_min = torch.cummin(second_diff, dim=0)[0]
        mask = (second_diff[1:] > 0) & (cum_min[:-1] < 0)
        transition_indices = torch.nonzero(mask, as_tuple=True)[0]

        if transition_indices.numel() > 0:
            # Adjust index by 1 because we used second_diff[1:]
            threshold = transition_indices[0].item() + 1
            threshold += 1  # Adjust for the first diff
            threshold += cut_index  # Adjust for the cut index
            return threshold

        else: # No spike found. Use `largest_gap` instead.
            return self.find_threshold_largest_gap(arr)

    def find_threshold(self, arr: torch.Tensor) -> int:
        """Find the threshold.
        Args:
            arr (torch.Tensor): A list of cosine similarities.
        Returns:
            int: The index where the largest numerical gap occurs.
        """
        if len(arr) < 2:
            raise ValueError("The list must contain at least two elements.")

        if isinstance(arr, torch.Tensor):
            pass
        elif isinstance(arr, list):
            arr = torch.tensor(arr)
        elif isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        else:
            raise TypeError(
                "Invalid type; expected either list, torch.Tensor, or numpy.ndarray.")

        if self._is_ascending(arr): # make it descending (flip it)
            arr = torch.flip(arr, dims=[0])
        elif self._is_descending(arr):
            pass
        else:
            raise ValueError("`num_list` is not sorted.")

        # Calculate the gaps between consecutive elements
        if self.strategy == "largest_gap":
            return self.find_threshold_largest_gap(arr)

        elif self.strategy == "moving_avg":
            return self.find_threshold_moving_avg(arr)

        elif self.strategy == "2diff_spike":
            return self.find_threshold_2diff_spike(arr)

        else:
            raise NotImplementedError

    def _visualize_distribution(self,
                                distribution: List[float] | torch.Tensor,
                                chart_type: Literal["bar", "line"] = "line",
                                outfile: str = "distribution.png") -> None:
        """Visualize a distribution in a bar chart or line graph.
        Make sure that distribution is sorted.

        Args:
            distribution (list): A list of cosine similarities.
            chart_type (str): Type of chart. Either `bar` or `line`.
            outfile (str): Output file name.
        """
        if isinstance(distribution, torch.Tensor):
            distribution = distribution.cpu().numpy()

        plt.figure(figsize=(10, 6))

        if chart_type == "bar":
            plt.bar(
                range(len(distribution)),
                distribution,
                color="skyblue",
                edgecolor="black"
            )

            plt.xlabel("Index", fontsize=12)
            plt.ylabel("Cosine Similarity", fontsize=12)
            plt.title("Cosine Similarities", fontsize=14)

        elif chart_type == "line":
            plt.plot(
                range(len(distribution)),
                distribution,
                color="blue",
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=4
            )

            plt.xlabel("Index", fontsize=12)
            plt.ylabel("Cosine Similarity", fontsize=12)
            plt.title("Cosine Similarities", fontsize=14)

        else:
            raise ValueError("Invalid chart type. Use `bar` or `line`.")

        # Grid and layout
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # save
        plt.savefig(outfile)

    def retrieve(self,
                 query: str,
                 context: list | List[Dict[str, int | str]],
                 sample: Optional[Dict[str, str]] = None,
                 adaptive: bool = False,
                 visualize: bool = False,
                 return_true_k: bool = False) -> Tuple[List[ContextType], int | None]:
        """Retrieve chunks based on the query and context.
        
        Args:
            query (str): The query text.
            context (list): A list of chunks.
            sample (dict): A sample chunk: {
                    "query": ins["question"],
                    "label": gold_answer,
                    "context": context,
                    "positive_context": positive context
                    "relevant_context": hard-negative context
                    }
            adaptive (bool): If True, use adaptive retrieval.
            visualize (bool): If True, visualize the distribution of cosine similarities.
            return_true_k (bool): If True, return the true k.
        Returns:
            List[ContextType]: A list of retrieved chunks.
            int | None: The true k.
        """
        if isinstance(context, list) and isinstance(context[0], dict):
            # QA type
            context_chunks = [chunk["text"] for chunk in context]

        if self.use_faiss:
            assert sample["idx"] is not None, \
                "sample['idx'] must be specified when using FAISS."
            # Load the FAISS index
            model_name = self.model.split("/")[-1]
            if self.task == "holobench":
                assert self.info_amount is not None, \
                    "info_amount must be specified when using FAISS."
                n_sample = str(int(self.info_amount // 1000)) + "k"
                index_file = f"{base_dir}vector_dbs/{self.task}_{n_sample}/{model_name}/{self.task}_{n_sample}_contexts_{sample['idx']}.index"
            else:
                index_file = f"{base_dir}vector_dbs/{self.task}/{model_name}/{self.task}_contexts_{sample['idx']}.index"
            if not os.path.exists(index_file):
                raise FileNotFoundError(
                    f"The specified index file {index_file} was not found."
                )
            # Load the FAISS index
            # print("Using the pre-made FAISS index:", index_file)
            index = faiss.read_index(index_file)
            if self.use_gpu:
                index = faiss.index_cpu_to_gpu(self.res, 0, index)

            query_emb = self.embedding.get_embedding(query,
                                                     batch_size=self.batch_size,
                                                     convert_to_numpy=True)
            # print("Query embeddings created.")
            # Get everything (`index.ntotal`) first
            D, I = index.search(query_emb,
                                k=index.ntotal)
            cos_sims = D[0] # already sorted in descending order
            indices = I[0]
        else:
            query_emb = self.embedding.get_embedding(query)
            # print("Query embeddings created.")
            if self.use_sentence_transformer:
                context_embs = self.embedding.get_embedding(
                    context_chunks,
                    batch_size=self.batch_size
                )
            else:
                raise NotImplementedError("Use sentence_transformers.")

            if self.compute_cossim_from_scratch:
                query_emb = query_emb.to(self.device)
                context_embs = context_embs.to(self.device)

                cos_sims: torch.Tensor = self.embedding.model.similarity(
                    query_emb, context_embs
                )
                # SentenceTransformer.similarity() uses cosine similarity by default
                cos_sims, indices = cos_sims.sort(descending=True)
                # torch.sort returns Tuple[tensor, tensor]: (sorted_tensor, original_indices)
                cos_sims = cos_sims[0]
                indices = indices[0]
            else: # compute the cosine similarity with FAISS
                index = faiss.IndexFlatIP(query_emb.shape[1])
                index.add(context_embs.cpu().numpy())
                D, I = index.search(query_emb.cpu().numpy(), k=index.ntotal)
                cos_sims = D[0]  # already sorted in descending order
                indices = I[0]


        true_k = None
        if return_true_k:
            positive_context_ids = set(
                    [ctx["psg_id"] for ctx in sample["positive_context"]]
                )
            for i, idx in enumerate(indices):
                if context[idx]["psg_id"] in positive_context_ids:
                    # sample["positive_context"] is a list of positive contexts:
                    # [{'psg_id': int, 'title': str, 'text': str}]
                    true_k = i

        if adaptive:
            retrieved_chunks = self.adaptive_retrieve(cos_sims,
                                                      indices,
                                                      context,)
        else:
            retrieved_chunks = self.fixed_retrieve(cos_sims,
                                                  indices,
                                                  context,)
        
        if visualize:
            self._visualize_distribution(cos_sims)
            
        return retrieved_chunks, true_k

    def adaptive_retrieve(self,
                          cos_sims: torch.Tensor | np.ndarray,
                          indices: torch.Tensor | np.ndarray,
                          context: list | List[Dict[str, int | str]]) -> List[ContextType]:
        """Adaptive retrieval.
        The number of retrieved chunks (k) will be determined dynamically
        based on the distribution of cosine similarities.

        Args:
            cos_sims (torch.Tensor | np.ndarray): Cosine similarities.
            indices (torch.Tensor | np.ndarray): Indices of the context chunks.
            context (list | List[Dict[str, int | str]]): A list of chunks.

        Returns:
            List[Dict[str, str | float]] | None: A list of retrieved chunks.
        """
        threshold = self.find_threshold(cos_sims)

        retrieved_chunks: List[dict] = []

        if self.retrieve_more:
            if isinstance(self.retrieve_more, float):
                threshold = int(threshold * self.retrieve_more)
            elif isinstance(self.retrieve_more, int):
                threshold = threshold + self.retrieve_more

        print(f"Threshold: {threshold}")

        # print(f"Top {threshold} matching chunks:")
        if isinstance(context, list):
            if isinstance(context[0], dict):
                # qa
                for i, (cos_sim, idx) in enumerate(zip(cos_sims, indices)):
                    if i > threshold:
                        break

                    if context[idx]["psg_id"] is not None:
                        psg_id = int(context[idx]["psg_id"])
                    elif context[idx]["id"] is not None:
                        psg_id = int(context[idx]["id"])
                    else:
                        psg_id = None

                    ret = {"psg_id": psg_id,
                           "chunk": context[idx]["text"],
                           "score": cos_sim}
                    retrieved_chunks.append(ret)
                    
            elif isinstance(context[0], str): # holobench
                for i, (cos_sim, idx) in enumerate(zip(cos_sims, indices)):
                    if i > threshold:
                        break
                    ret = {"psg_id": int(idx),
                           "chunk": context[idx],
                           "score": cos_sim}
                    retrieved_chunks.append(ret)

        return retrieved_chunks

    def fixed_retrieve(self,
                       cos_sims: torch.Tensor | np.ndarray,
                       indices: torch.Tensor | np.ndarray,
                       context: list | List[Dict[str, int | str]]) -> List[ContextType]:
        """Retrieve a fixed number of chunks.
        Args:
            cos_sims (torch.Tensor | np.ndarray): Cosine similarities.
            indices (torch.Tensor | np.ndarray): Indices of the context chunks.
            context (list | List[Dict[str, int | str]]): A list of chunks.
        Returns:
            List[Dict[str, str | float]]: A list of retrieved chunks.
        """
        token_length = 0
        retrieved_chunks = []

        if isinstance(context, list):
            if isinstance(context[0], dict):
                for i, (cos_sim, idx) in enumerate(zip(cos_sims, indices)):
                    if self.fixed_retrieval_k:
                        if i == self.fixed_retrieval_k: # enough context chunks
                            break
                    elif self.fixed_retrieval_tokens:
                        # rough estimate
                        token_length += len(context[idx]["text"]) // self.char_to_token_ratio
                        if token_length >= self.fixed_retrieval_tokens:
                            break
                        
                    if context[idx]["psg_id"] is not None:
                        psg_id = int(context[idx]["psg_id"])
                    elif context[idx]["id"] is not None:
                        psg_id = int(context[idx]["id"])
                    else:
                        psg_id = None
                    
                    # print("psg_id:", psg_id)
                    ret = {"psg_id": psg_id,
                           "chunk": context[idx]["text"],
                           "score": cos_sim}
                    
                    retrieved_chunks.append(ret)

            elif isinstance(context[0], str):
                for i, (cos_sim, idx) in enumerate(zip(cos_sims, indices)):
                    if self.fixed_retrieval_k:
                        if i == self.fixed_retrieval_k:
                            break
                    elif self.fixed_retrieval_tokens:
                        token_length += len(context[idx]["text"]) // self.char_to_token_ratio
                        if token_length >= self.fixed_retrieval_tokens:
                            break

                    ret = {"psg_id": int(idx),
                           "chunk": context[idx],
                           "score": cos_sim}
                    
                    retrieved_chunks.append(ret)
                    
        else:
            raise TypeError(
                f"Expected a list of chunks, but got {type(context)}."
            )

        return retrieved_chunks
